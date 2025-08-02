package com.example.frontend.utils

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.util.Log
import android.util.Size
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer

object CameraUtils {
    fun getSupportedResolutions(context: Context): List<Size> {
        val resolutions = mutableListOf<Size>()
        try {
            val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraIds = cameraManager.cameraIdList

            for (cameraId in cameraIds) {
                val characteristics = cameraManager.getCameraCharacteristics(cameraId)
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)

                if (facing == CameraCharacteristics.LENS_FACING_BACK) {
                    val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                    map?.getOutputSizes(android.graphics.ImageFormat.JPEG)?.let { sizes ->
                        resolutions.addAll(sizes.map { Size(it.width, it.height) })
                    }
                    break
                }
            }
        } catch (e: Exception) {
            Log.e("CameraUtils", "Getting Supported Resolutions Failed", e)
        }

        return resolutions.distinctBy { "${it.width}x${it.height}" }
            .sortedByDescending { it.width * it.height }
    }

    fun imageProxyToByteArray(image: ImageProxy): ByteArray {
        val buffer: ByteBuffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return bytes
    }

    fun getRecommendedCaptureSize(context: Context): Size {
        val supportedSizes = getSupportedResolutions(context)

        val preferredSizes = supportedSizes.filter { size ->
            val ratio = size.width.toFloat() / size.height.toFloat()
            val megapixels = (size.width * size.height) / 1_000_000.0

            kotlin.math.abs(ratio - 4f/3f) < 0.1f && megapixels in 2.0..12.0
        }

        return preferredSizes.firstOrNull() ?: Size(1920, 1440)
    }

    fun hasBackCamera(context: Context): Boolean {
        return try {
            val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraIds = cameraManager.cameraIdList

            for (cameraId in cameraIds) {
                val characteristics = cameraManager.getCameraCharacteristics(cameraId)
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (facing == CameraCharacteristics.LENS_FACING_BACK) {
                    return true
                }
            }
            false
        } catch (e: Exception) {
            Log.e("CameraUtils", "Checking back camera failed", e)
            false
        }
    }

    fun hasFrontCamera(context: Context): Boolean {
        return try {
            val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraIds = cameraManager.cameraIdList

            for (cameraId in cameraIds) {
                val characteristics = cameraManager.getCameraCharacteristics(cameraId)
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    return true
                }
            }
            false
        } catch (e: Exception) {
            Log.e("CameraUtils", "Checking front camera failed", e)
            false
        }
    }
}