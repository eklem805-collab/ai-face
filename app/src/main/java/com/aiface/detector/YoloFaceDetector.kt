package com.aiface.detector

import android.content.Context
import android.graphics.*
import android.util.Log
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/**
 * YOLOv8 Face Detector using TensorFlow Lite
 * Supports yolov8n-face.tflite and yolov8l-face.tflite
 *
 * Model input: 640x640 RGB, output: [1, 4+1+5*2] or similar (boxes + conf + keypoints)
 * For yolov8-face: output format is [1, 6+10, N] -> x,y,w,h,conf, kpts...
 *
 * This implementation handles both:
 * - Standard YOLOv8 face: 5 keypoints (eyes, nose, mouth corners) = 10 values
 * - If model only outputs boxes, we just draw boxes
 */
class YoloFaceDetector(private val context: Context) {

    companion object {
        private const val TAG = "YoloFaceDetector"
        private const val INPUT_SIZE = 640
        private const val CONF_THRESHOLD = 0.5f
        private const val IOU_THRESHOLD = 0.45f
    }

    private var interpreter: Interpreter? = null
    private var isInitialized = false

    fun initialize(modelPath: String = "yolov8n-face.tflite"): Boolean {
        // Try all possible model names
        val modelsToTry = listOf(modelPath, "yolov8n-face.tflite", "yolov8l-face.tflite", "yolo-face.tflite", "yolov8-face.tflite")
        for (mp in modelsToTry.distinct()) {
            try {
                Log.i(TAG, "Trying to load YOLO model: $mp")
                val model = FileUtil.loadMappedFile(context, mp)
                val options = Interpreter.Options().apply {
                    setNumThreads(4)
                    // Don't use NNAPI for compatibility - some devices fail with NNAPI
                    setUseNNAPI(false)
                }
                interpreter = Interpreter(model, options)
                isInitialized = true
                Log.i(TAG, "YOLO model loaded SUCCESS: $mp size ${model.capacity()}")
                Log.i(TAG, "Input tensor: ${interpreter?.getInputTensor(0)?.shape()?.contentToString()}")
                Log.i(TAG, "Output tensor: ${interpreter?.getOutputTensor(0)?.shape()?.contentToString()}")
                return true
            } catch (e: Exception) {
                Log.w(TAG, "Failed to load $mp: ${e.message}")
            }
        }
        Log.e(TAG, "All YOLO models failed to load, will use ML Kit")
        return false
    }

    data class YoloResult(
        val box: RectF,
        val conf: Float,
        val keypoints: List<PointF>
    )

    fun detect(imageProxy: ImageProxy): List<DetectedFace> {
        if (!isInitialized || interpreter == null) return emptyList()

        val bitmap = imageProxyToBitmap(imageProxy) ?: return emptyList()
        val inputSize = INPUT_SIZE

        // Resize and pad to 640x640 (letterbox)
        val (resizedBitmap, scale, dx, dy, origW, origH) = letterbox(bitmap, inputSize)

        // Prepare input buffer
        val inputBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)
        for (pixel in intValues) {
            // RGB, normalized 0-1
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
            inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
            inputBuffer.putFloat((pixel and 0xFF) / 255f)
        }

        // Output - shape depends on model, try to infer
        val outputTensor = interpreter!!.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        Log.d(TAG, "Output shape: ${outputShape.contentToString()}")

        // Typical YOLOv8 face output: [1, 16, 8400] or [1, 8400, 16] or [1, 6, 8400]
        // 16 = 4 box + 1 conf + 10 kpts (5 points) + 1 extra?
        // Let's handle generic case
        val results = mutableListOf<YoloResult>()

        try {
            if (outputShape.size == 3) {
                val dim1 = outputShape[1]
                val dim2 = outputShape[2]
                // Two possible layouts
                if (dim1 < dim2) {
                    // [1, 16, 8400] -> transpose
                    val output = Array(1) { Array(dim1) { FloatArray(dim2) } }
                    interpreter!!.run(inputBuffer, output)
                    // Parse as [16][8400]
                    for (i in 0 until dim2) {
                        val x = output[0][0][i]
                        val y = output[0][1][i]
                        val w = output[0][2][i]
                        val h = output[0][3][i]
                        val conf = if (dim1 > 4) output[0][4][i] else 1f

                        if (conf < CONF_THRESHOLD) continue

                        val left = x - w / 2
                        val top = y - h / 2
                        val right = x + w / 2
                        val bottom = y + h / 2

                        // Keypoints if available
                        val kpts = mutableListOf<PointF>()
                        if (dim1 >= 15) {
                            // 5 keypoints: 10 values
                            for (k in 0 until 5) {
                                val kx = output[0][5 + k * 2][i]
                                val ky = output[0][5 + k * 2 + 1][i]
                                kpts.add(PointF(kx, ky))
                            }
                        }

                        results.add(YoloResult(RectF(left, top, right, bottom), conf, kpts))
                    }
                } else {
                    // [1, 8400, 16]
                    val output = Array(1) { Array(dim1) { FloatArray(dim2) } }
                    interpreter!!.run(inputBuffer, output)
                    for (i in 0 until dim1) {
                        val row = output[0][i]
                        if (row.size < 5) continue
                        val x = row[0]
                        val y = row[1]
                        val w = row[2]
                        val h = row[3]
                        val conf = row[4]
                        if (conf < CONF_THRESHOLD) continue
                        val left = x - w / 2
                        val top = y - h / 2
                        val right = x + w / 2
                        val bottom = y + h / 2

                        val kpts = mutableListOf<PointF>()
                        if (row.size >= 15) {
                            for (k in 0 until 5) {
                                val kx = row[5 + k * 2]
                                val ky = row[5 + k * 2 + 1]
                                kpts.add(PointF(kx, ky))
                            }
                        }
                        results.add(YoloResult(RectF(left, top, right, bottom), conf, kpts))
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}", e)
        }

        // NMS
        val filtered = nms(results)

        // Convert back from letterboxed 640x640 to original image coordinates
        // Original image is imageProxy width/height
        return filtered.map { res ->
            // Undo letterbox
            var box = RectF(
                (res.box.left - dx) / scale,
                (res.box.top - dy) / scale,
                (res.box.right - dx) / scale,
                (res.box.bottom - dy) / scale
            )
            // Clamp to original bitmap
            box.left = max(0f, min(box.left, origW.toFloat()))
            box.top = max(0f, min(box.top, origH.toFloat()))
            box.right = max(0f, min(box.right, origW.toFloat()))
            box.bottom = max(0f, min(box.bottom, origH.toFloat()))

            // Keypoints also need to be transformed
            val kptsTransformed = res.keypoints.map { kp ->
                PointF(
                    (kp.x - dx) / scale,
                    (kp.y - dy) / scale
                )
            }

            // Scale from bitmap size to imageProxy size (they should be same, but ensure)
            // imageProxy width/height might be different from bitmap due to rotation
            // For simplicity, we assume bitmap is already rotated correctly

            DetectedFace(
                boundingBox = box,
                confidence = res.conf,
                landmarks = kptsTransformed,
                contours = emptyList()
            )
        }
    }

    private fun letterbox(
        bitmap: Bitmap,
        inputSize: Int
    ): LetterboxResult {
        val origW = bitmap.width
        val origH = bitmap.height
        val scale = min(inputSize.toFloat() / origW, inputSize.toFloat() / origH)
        val newW = (origW * scale).toInt()
        val newH = (origH * scale).toInt()

        val resized = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
        val result = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        canvas.drawColor(Color.BLACK)
        val dx = (inputSize - newW) / 2f
        val dy = (inputSize - newH) / 2f
        canvas.drawBitmap(resized, dx, dy, null)

        return LetterboxResult(result, scale, dx, dy, origW, origH)
    }

    data class LetterboxResult(
        val bitmap: Bitmap,
        val scale: Float,
        val dx: Float,
        val dy: Float,
        val origW: Int,
        val origH: Int
    )

    private fun nms(results: List<YoloResult>): List<YoloResult> {
        val sorted = results.sortedByDescending { it.conf }.toMutableList()
        val keep = mutableListOf<YoloResult>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            keep.add(best)
            val iterator = sorted.iterator()
            while (iterator.hasNext()) {
                val other = iterator.next()
                if (iou(best.box, other.box) > IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }
        return keep
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)

        val interW = max(0f, interRight - interLeft)
        val interH = max(0f, interBottom - interTop)
        val interArea = interW * interH

        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        val union = areaA + areaB - interArea

        return if (union <= 0) 0f else interArea / union
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        // YUV to RGB conversion
        val buffer = imageProxy.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        // For simplicity, if format is already RGB, but ImageProxy is YUV_420_888
        // We'll do a simple conversion using Bitmap
        try {
            val yBuffer = imageProxy.planes[0].buffer
            val uBuffer = imageProxy.planes[1].buffer
            val vBuffer = imageProxy.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)

            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = android.graphics.YuvImage(
                nv21,
                android.graphics.ImageFormat.NV21,
                imageProxy.width,
                imageProxy.height,
                null
            )
            val out = java.io.ByteArrayOutputStream()
            yuvImage.compressToJpeg(
                android.graphics.Rect(0, 0, imageProxy.width, imageProxy.height),
                100,
                out
            )
            val imageBytes = out.toByteArray()
            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        } catch (e: Exception) {
            Log.e(TAG, "YUV to Bitmap failed: ${e.message}")
            return null
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
        isInitialized = false
    }
}
