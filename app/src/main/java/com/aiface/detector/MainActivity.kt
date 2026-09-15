package com.aiface.detector

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.aiface.detector.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var preview: Preview? = null

    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private var detectionMode = DetectionMode.MLKIT

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var mlkitHelper: FaceDetectorHelper
    private var yoloHelper: YoloFaceDetector? = null

    private var lastFpsTime = System.currentTimeMillis()
    private var frameCount = 0
    private var fps = 0f

    enum class DetectionMode {
        MLKIT, YOLO
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Камера нужна для детекции", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        mlkitHelper = FaceDetectorHelper(this)

        // Try init YOLO
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val helper = YoloFaceDetector(this@MainActivity)
                val ok = helper.initialize("yolov8n-face.tflite")
                if (ok) {
                    yoloHelper = helper
                    Log.i("MainActivity", "YOLO initialized")
                } else {
                    Log.w("MainActivity", "YOLO init failed, will use MLKit only")
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "YOLO error: ${e.message}")
            }
        }

        binding.btnSwitchCamera.setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                CameraSelector.LENS_FACING_BACK
            } else {
                CameraSelector.LENS_FACING_FRONT
            }
            startCamera()
        }

        binding.btnToggleMode.setOnClickListener {
            detectionMode = if (detectionMode == DetectionMode.MLKIT) {
                if (yoloHelper == null) {
                    Toast.makeText(this, "YOLO модель не загружена, используется ML Kit", Toast.LENGTH_SHORT).show()
                    DetectionMode.MLKIT
                } else {
                    DetectionMode.YOLO
                }
            } else {
                DetectionMode.MLKIT
            }
            binding.modeText.text = when (detectionMode) {
                DetectionMode.MLKIT -> "ML Kit"
                DetectionMode.YOLO -> "YOLOv8"
            }
        }

        // Check permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: return

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    processImageProxy(imageProxy)
                }
            }

        provider.unbindAll()

        try {
            camera = provider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )
            preview?.setSurfaceProvider(binding.previewView.surfaceProvider)
        } catch (e: Exception) {
            Log.e("MainActivity", "Bind failed", e)
        }
    }

    private fun processImageProxy(imageProxy: ImageProxy) {
        val start = System.currentTimeMillis()

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val faces = when (detectionMode) {
                    DetectionMode.MLKIT -> mlkitHelper.detect(imageProxy)
                    DetectionMode.YOLO -> yoloHelper?.detect(imageProxy) ?: mlkitHelper.detect(imageProxy)
                }

                // FPS calc
                frameCount++
                val now = System.currentTimeMillis()
                if (now - lastFpsTime > 1000) {
                    fps = frameCount * 1000f / (now - lastFpsTime)
                    frameCount = 0
                    lastFpsTime = now
                }

                // Update UI
                launch(Dispatchers.Main) {
                    binding.overlayView.setResults(
                        faces,
                        imageProxy.width,
                        imageProxy.height,
                        lensFacing == CameraSelector.LENS_FACING_FRONT
                    )
                    binding.fpsText.text = String.format("FPS: %.1f", fps)
                    binding.facesText.text = "Лиц: ${faces.size}"
                }

            } catch (e: Exception) {
                Log.e("MainActivity", "Detection error: ${e.message}", e)
            } finally {
                imageProxy.close()
                val elapsed = System.currentTimeMillis() - start
                if (elapsed > 100) {
                    Log.d("MainActivity", "Slow frame: ${elapsed}ms")
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        mlkitHelper.close()
        yoloHelper?.close()
    }
}
