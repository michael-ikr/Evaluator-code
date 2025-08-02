package com.example.frontend.utils

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

object MediaUtils {

    suspend fun saveVideoToGallery(context: Context, videoUri: Uri): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                val inputStream = context.contentResolver.openInputStream(videoUri)
                val videoBytes = inputStream?.readBytes()
                inputStream?.close()

                if (videoBytes == null) {
                    Log.e("MediaUtils", "Cannot read file")
                    return@withContext false
                }

                val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
                val fileName = "VIDEO_$timeStamp.mp4"

                val contentValues = ContentValues().apply {
                    put(MediaStore.Video.Media.DISPLAY_NAME, fileName)
                    put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                    put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES)
                }

                val uri = context.contentResolver.insert(
                    MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                    contentValues
                )

                uri?.let { outputUri ->
                    context.contentResolver.openOutputStream(outputUri)?.use { outputStream ->
                        outputStream.write(videoBytes)
                        outputStream.flush()
                    }
                    Log.i("MediaUtils", "Video saved to gallery: $fileName")
                    true
                } ?: false

            } catch (e: Exception) {
                Log.e("MediaUtils", "Failed to save video to gallery", e)
                false
            }
        }
    }

    fun isVideoFile(context: Context, uri: Uri): Boolean {
        return try {
            val mimeType = context.contentResolver.getType(uri)
            mimeType?.startsWith("video/") == true
        } catch (e: Exception) {
            false
        }
    }
}