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

    // PURPLE PIXEL STYLE - for live camera
    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = Color.parseColor("#E0AAFF") // purple glow
        isAntiAlias = true
        // Glow effect
        setShadowLayer(15f, 0f, 0f, Color.parseColor("#9D4EDD"))
    }

    private val boxFillPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.argb(40, 157, 78, 221) // purple transparent
    }

    private val boxInnerPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
        color = Color.parseColor("#7B2CBF")
        pathEffect = DashPathEffect(floatArrayOf(12f, 6f), 0f)
        isAntiAlias = true
    }

    private val cornerPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = Color.parseColor("#FF79C6") // pink neon corners
        isAntiAlias = true
        strokeCap = Paint.Cap.SQUARE
    }

    private val landmarkPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#FF79C6") // pink neon dots
        isAntiAlias = true
        setShadowLayer(8f, 0f, 0f, Color.parseColor("#FF79C6"))
    }

    private val contourPaintBase = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 2.5f
        color = Color.argb(150, 157, 78, 221) // purple contour
        isAntiAlias = true
    }

    private var pixelTypeface: Typeface? = null

    private val textPaint = Paint().apply {
        color = Color.parseColor("#E0AAFF")
        textSize = 28f
        isAntiAlias = true
        setShadowLayer(4f, 2f, 2f, Color.parseColor("#4A148C"))
    }

    private val textBgPaint = Paint().apply {
        color = Color.argb(230, 26, 11, 46) // dark purple bg
        style = Paint.Style.FILL
    }

    private val textBorderPaint = Paint().apply {
        color = Color.parseColor("#9D4EDD")
        style = Paint.Style.STROKE
        strokeWidth = 3f
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

            // Load pixel font if not loaded
            if (pixelTypeface == null) {
                try {
                    pixelTypeface = Typeface.createFromAsset(context.assets, "PressStart2P-Regular.ttf")
                    textPaint.typeface = pixelTypeface
                } catch (e: Exception) {
                    // fallback to default
                }
            }

            // Draw filled background with purple
            canvas.drawRoundRect(mappedRect, 12f, 12f, boxFillPaint)
            // Draw inner dashed purple
            canvas.drawRoundRect(mappedRect, 12f, 12f, boxInnerPaint)
            // Draw outer glowing border
            canvas.drawRoundRect(mappedRect, 12f, 12f, boxPaint)

            // Draw corner brackets - PIXEL STYLE PURPLE
            val bracketLen = 32f
            // Top-left
            canvas.drawLine(mappedRect.left, mappedRect.top + bracketLen, mappedRect.left, mappedRect.top, cornerPaint)
            canvas.drawLine(mappedRect.left, mappedRect.top, mappedRect.left + bracketLen, mappedRect.top, cornerPaint)
            // Top-right
            canvas.drawLine(mappedRect.right - bracketLen, mappedRect.top, mappedRect.right, mappedRect.top, cornerPaint)
            canvas.drawLine(mappedRect.right, mappedRect.top, mappedRect.right, mappedRect.top + bracketLen, cornerPaint)
            // Bottom-left
            canvas.drawLine(mappedRect.left, mappedRect.bottom - bracketLen, mappedRect.left, mappedRect.bottom, cornerPaint)
            canvas.drawLine(mappedRect.left, mappedRect.bottom, mappedRect.left + bracketLen, mappedRect.bottom, cornerPaint)
            // Bottom-right
            canvas.drawLine(mappedRect.right - bracketLen, mappedRect.bottom, mappedRect.right, mappedRect.bottom, cornerPaint)
            canvas.drawLine(mappedRect.right, mappedRect.bottom - bracketLen, mappedRect.right, mappedRect.bottom, cornerPaint)

            // Draw confidence with pixel font
            val label = String.format("FACE %.0f%%", face.confidence * 100)
            val textWidth = textPaint.measureText(label)
            val textBgRect = RectF(
                mappedRect.left,
                mappedRect.top - 48f,
                mappedRect.left + textWidth + 28f,
                mappedRect.top - 4f
            )
            canvas.drawRoundRect(textBgRect, 6f, 6f, textBgPaint)
            canvas.drawRoundRect(textBgRect, 6f, 6f, textBorderPaint)
            canvas.drawText(label, mappedRect.left + 14f, mappedRect.top - 16f, textPaint)

            // Draw landmarks (eyes, nose, etc) as PURPLE PIXEL polygons
            for (point in face.landmarks) {
                var px = point.x * scale + dx
                var py = point.y * scale + dy
                if (isFrontCamera) {
                    px = viewWidth - px
                }
                // Only every 2nd point for pixel effect
                canvas.drawCircle(px, py, 6f, landmarkPaint)
            }

            // Draw face contours / polygons - PURPLE
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
                if (contour.size > 8) path.close()
                canvas.drawPath(path, contourPaintBase)
            }
        }
    }
}
