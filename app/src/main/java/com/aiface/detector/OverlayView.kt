package com.aiface.detector

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

data class DetectedFace(
    val boundingBox: RectF,
    val confidence: Float = 1f,
    val landmarks: List<PointF> = emptyList(), // eyes, nose, mouth etc
    val contours: List<List<PointF>> = emptyList() // face polygon
)

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var faces: List<DetectedFace> = emptyList()
    private var imageWidth: Int = 0
    private var imageHeight: Int = 0
    private var isFrontCamera: Boolean = false

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = Color.GREEN
        isAntiAlias = true
    }

    private val boxFillPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.argb(40, 0, 255, 0)
    }

    private val landmarkPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.MAGENTA
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 38f
        typeface = Typeface.DEFAULT_BOLD
        isAntiAlias = true
        setShadowLayer(4f, 2f, 2f, Color.BLACK)
    }

    private val textBgPaint = Paint().apply {
        color = Color.argb(180, 0, 0, 0)
        style = Paint.Style.FILL
    }

    fun setResults(
        faces: List<DetectedFace>,
        imageWidth: Int,
        imageHeight: Int,
        isFrontCamera: Boolean
    ) {
        this.faces = faces
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight
        this.isFrontCamera = isFrontCamera
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (imageWidth == 0 || imageHeight == 0) return

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()

        // Scale to fit centerCrop style like PreviewView
        val scale: Float
        val dx: Float
        val dy: Float

        val scaleX = viewWidth / imageWidth
        val scaleY = viewHeight / imageHeight
        scale = max(scaleX, scaleY)
        val scaledWidth = imageWidth * scale
        val scaledHeight = imageHeight * scale
        dx = (viewWidth - scaledWidth) / 2f
        dy = (viewHeight - scaledHeight) / 2f

        // Transform matrix for bounding boxes
        for (face in faces) {
            val rect = face.boundingBox

            // Transform from image coordinates to view coordinates
            var left = rect.left * scale + dx
            var top = rect.top * scale + dy
            var right = rect.right * scale + dx
            var bottom = rect.bottom * scale + dy

            if (isFrontCamera) {
                // Mirror horizontally for front camera
                val mirroredLeft = viewWidth - right
                val mirroredRight = viewWidth - left
                left = mirroredLeft
                right = mirroredRight
            }

            val mappedRect = RectF(left, top, right, bottom)

            // Draw filled background
            canvas.drawRoundRect(mappedRect, 16f, 16f, boxFillPaint)
            // Draw border
            canvas.drawRoundRect(mappedRect, 16f, 16f, boxPaint)

            // Draw confidence
            val label = String.format("%.0f%%", face.confidence * 100)
            val textWidth = textPaint.measureText(label)
            val textBgRect = RectF(
                mappedRect.left,
                mappedRect.top - 50f,
                mappedRect.left + textWidth + 24f,
                mappedRect.top
            )
            canvas.drawRoundRect(textBgRect, 8f, 8f, textBgPaint)
            canvas.drawText(label, mappedRect.left + 12f, mappedRect.top - 12f, textPaint)

            // Draw landmarks (eyes, nose, etc) as polygons
            for (point in face.landmarks) {
                var px = point.x * scale + dx
                var py = point.y * scale + dy
                if (isFrontCamera) {
                    px = viewWidth - px
                }
                canvas.drawCircle(px, py, 8f, landmarkPaint)
            }

            // Draw face contours / polygons
            val contourPaint = Paint().apply {
                style = Paint.Style.STROKE
                strokeWidth = 3f
                color = Color.CYAN
                isAntiAlias = true
            }
            for (contour in face.contours) {
                if (contour.size < 2) continue
                val path = Path()
                var first = true
                for (p in contour) {
                    var px = p.x * scale + dx
                    var py = p.y * scale + dy
                    if (isFrontCamera) px = viewWidth - px
                    if (first) {
                        path.moveTo(px, py)
                        first = false
                    } else {
                        path.lineTo(px, py)
                    }
                }
                // close if it's a face oval
                if (contour.size > 8) path.close()
                canvas.drawPath(path, contourPaint)
            }
        }
    }
}
