package com.aiface.detector

import android.content.Context
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.RectF
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import kotlinx.coroutines.tasks.await

class FaceDetectorHelper(private val context: Context) {

    private val detector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .setMinFaceSize(0.15f)
            .enableTracking()
            .build()
        FaceDetection.getClient(options)
    }

    suspend fun detect(imageProxy: ImageProxy): List<DetectedFace> {
        val mediaImage = imageProxy.image ?: return emptyList()
        val rotation = imageProxy.imageInfo.rotationDegrees
        val inputImage = InputImage.fromMediaImage(mediaImage, rotation)

        return try {
            val faces = detector.process(inputImage).await()
            faces.map { face ->
                val bounds: Rect = face.boundingBox
                val box = RectF(
                    bounds.left.toFloat(),
                    bounds.top.toFloat(),
                    bounds.right.toFloat(),
                    bounds.bottom.toFloat()
                )

                // Landmarks: eyes, nose, mouth etc
                val landmarks = mutableListOf<PointF>()
                FaceLandmark.LEFT_EYE
                val types = listOf(
                    FaceLandmark.LEFT_EYE,
                    FaceLandmark.RIGHT_EYE,
                    FaceLandmark.NOSE_BASE,
                    FaceLandmark.LEFT_EAR,
                    FaceLandmark.RIGHT_EAR,
                    FaceLandmark.MOUTH_LEFT,
                    FaceLandmark.MOUTH_RIGHT,
                    FaceLandmark.MOUTH_BOTTOM
                )
                for (type in types) {
                    face.getLandmark(type)?.let {
                        landmarks.add(PointF(it.position.x, it.position.y))
                    }
                }

                // Contours - face oval, etc as polygons
                val contours = mutableListOf<List<PointF>>()
                face.allContours.forEach { contour ->
                    val points = contour.points.map { PointF(it.x, it.y) }
                    contours.add(points)
                }

                DetectedFace(
                    boundingBox = box,
                    confidence = 1f,
                    landmarks = landmarks,
                    contours = contours
                )
            }
        } catch (e: Exception) {
            e.printStackTrace()
            emptyList()
        }
    }

    fun close() {
        detector.close()
    }
}
