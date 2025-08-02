// CameraViewModel.kt
package com.example.frontend.viewmodel

import android.app.Application
import android.content.Context
import android.net.Uri
import android.util.Base64
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.net.NetworkInterface
import java.util.*

// Data types
data class Point(val x: Float, val y: Float)
data class Line(val start: Point, val end: Point)

data class CameraUiState(
    val isCameraOpen: Boolean = false,
    val isRecording: Boolean = false,
    val isLoading: Boolean = false,
    val points: List<Point> = emptyList(),
    val lines: List<Line> = emptyList(),
    val supination: String = "none",
    val selectedVideoUri: Uri? = null,
    val processedVideoUri: Uri? = null,
    val errorMessage: String? = null,
    val ipAddress: String = "",
    val isConnected: Boolean = false
)

class CameraViewModel(application: Application) : AndroidViewModel(application) {

    private val context: Context = application.applicationContext

    // TODO: change this line to your local IP address
    private val serverIP = "10.186.19.92"
    private val serverPort = 8000

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
        .readTimeout(0, java.util.concurrent.TimeUnit.SECONDS)
        .writeTimeout(0, java.util.concurrent.TimeUnit.SECONDS)
        .callTimeout(0, java.util.concurrent.TimeUnit.SECONDS)
        .build()

    private var webSocket: WebSocket? = null

    // UI state
    private val _uiState = MutableStateFlow(CameraUiState())
    val uiState: StateFlow<CameraUiState> = _uiState.asStateFlow()

    init {
        getLocalIpAddress()
    }

    private fun getLocalIpAddress() {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val interfaces = Collections.list(NetworkInterface.getNetworkInterfaces())
                for (networkInterface in interfaces) {
                    val addresses = Collections.list(networkInterface.inetAddresses)
                    for (address in addresses) {
                        if (!address.isLoopbackAddress && address.hostAddress?.contains(':') == false) {
                            _uiState.update { it.copy(ipAddress = address.hostAddress ?: "") }
                            return@launch
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e("NetworkUtils", "Failed to fetch IP address", e)
                _uiState.update { it.copy(ipAddress = "Unknown") }
            }
        }
    }

    fun openCamera() {
        _uiState.update {
            it.copy(
                isCameraOpen = true,
                selectedVideoUri = null,
                processedVideoUri = null,
                errorMessage = null
            )
        }
        connectWebSocket()
    }

    fun closeCamera() {
        stopRecording()
        disconnectWebSocket()
        _uiState.update {
            it.copy(
                isCameraOpen = false,
                isRecording = false,
                points = emptyList(),
                lines = emptyList(),
                supination = "none"
            )
        }
    }

    // Start/Stop Recording
    fun toggleRecording() {
        if (_uiState.value.isRecording) {
            stopRecording()
        } else {
            startRecording()
        }
    }

    private fun startRecording() {
        _uiState.update { it.copy(isRecording = true) }
    }

    private fun stopRecording() {
        _uiState.update {
            it.copy(
                isRecording = false,
                points = emptyList(),
                lines = emptyList()
            )
        }
    }

    // WebSocket connection management
    private fun connectWebSocket() {
        val request = Request.Builder()
            .url("ws://$serverIP:$serverPort/ws")
            .build()

        webSocket = httpClient.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d("WebSocket", "WebSocket Connected")
                _uiState.update { it.copy(isConnected = true, errorMessage = null) }
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val jsonObject = JSONObject(text)
                    processServerResponse(jsonObject)
                } catch (e: Exception) {
                    Log.e("WebSocket", "Message analysis failed", e)
                    _uiState.update { it.copy(errorMessage = "Message analysis failed: ${e.message}") }
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e("WebSocket", "WebSocket connection failed", t)
                _uiState.update {
                    it.copy(
                        isConnected = false,
                        errorMessage = "WebSocket connection failed: ${t.message}"
                    )
                }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.d("WebSocket", "WebSocket connection closed: $code - $reason")
                _uiState.update { it.copy(isConnected = false) }
            }
        })
    }

    private fun disconnectWebSocket() {
        webSocket?.close(1000, "WebSocket closed by user")
        webSocket = null
        _uiState.update { it.copy(isConnected = false) }
    }

    private fun processServerResponse(jsonObject: JSONObject) {
        viewModelScope.launch {
            try {
                val coordList = jsonObject.optJSONObject("coord_list")
                val classificationList = jsonObject.optJSONObject("classification_list")
                val supinationValue = jsonObject.optString("supination", "none")

                // update posture info
                val wristPosture = classificationList?.optString("wrist posture") ?: supinationValue

                val newPoints = mutableListOf<Point>()
                val newLines = mutableListOf<Line>()

                coordList?.let { coords ->
                    // extract points & boxes
                    val boxKeys = listOf(
                        "box bow top left", "box bow top right",
                        "box bow bottom right", "box bow bottom left",
                        "box string top left", "box string top right",
                        "box string bottom right", "box string bottom left"
                    )

                    val boxPoints = mutableListOf<Point>()
                    boxKeys.forEach { key ->
                        coords.optJSONArray(key)?.let { array ->
                            if (array.length() >= 2) {
                                val point = Point(
                                    array.getDouble(0).toFloat(),
                                    array.getDouble(1).toFloat()
                                )
                                newPoints.add(point)
                                boxPoints.add(point)
                            }
                        }
                    }

                    // Extract hand points
                    coords.optJSONArray("hand points")?.let { handPoints ->
                        for (i in 0 until handPoints.length()) {
                            handPoints.optJSONArray(i)?.let { point ->
                                if (point.length() >= 2) {
                                    newPoints.add(Point(
                                        point.getDouble(0).toFloat(),
                                        point.getDouble(1).toFloat()
                                    ))
                                }
                            }
                        }
                    }

                    if (boxPoints.size >= 8) {
                        for (i in 0..3) {
                            val start = boxPoints[i]
                            val end = boxPoints[(i + 1) % 4]
                            newLines.add(Line(start, end))
                        }

                        for (i in 4..7) {
                            val start = boxPoints[i]
                            val end = boxPoints[4 + ((i - 4 + 1) % 4)]
                            newLines.add(Line(start, end))
                        }
                    }
                }

                _uiState.update {
                    it.copy(
                        points = newPoints,
                        lines = newLines,
                        supination = wristPosture
                    )
                }

            } catch (e: Exception) {
                Log.e("ProcessResponse", "Failed to process response", e)
                _uiState.update { it.copy(errorMessage = "Failed to process response: ${e.message}") }
            }
        }
    }

    fun sendImageToServer(imageBytes: ByteArray) {
        if (webSocket?.request()?.url?.host == null || !_uiState.value.isConnected) {
            Log.w("SendImage", "WebSocket not connected")
            return
        }

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val base64String = Base64.encodeToString(imageBytes, Base64.NO_WRAP)
                val message = JSONObject().apply {
                    put("type", "frame")
                    put("image", "data:image/jpeg;base64,$base64String")
                }

                webSocket?.send(message.toString())
            } catch (e: Exception) {
                Log.e("SendImage", "Failed to send image", e)
                withContext(Dispatchers.Main) {
                    _uiState.update { it.copy(errorMessage = "Failed to send image: ${e.message}") }
                }
            }
        }
    }


    fun selectVideo(uri: Uri) {
        _uiState.update {
            it.copy(
                selectedVideoUri = uri,
                processedVideoUri = null,
                errorMessage = null
            )
        }
    }

    fun sendVideoToServer() {
        Log.d("VideoUpload", "=== Start sending video ===")
        val videoUri = _uiState.value.selectedVideoUri ?: run {
            Log.e("VideoUpload", "No video file selected")
            return
        }

        Log.d("VideoUpload", " Selected video URI: $videoUri")
        _uiState.update { it.copy(isLoading = true, errorMessage = null) }

        viewModelScope.launch(Dispatchers.IO) {
            try {
                Log.d("VideoUpload", "Start reading in video file")
                val inputStream = context.contentResolver.openInputStream(videoUri)
                val videoBytes = inputStream?.readBytes()
                inputStream?.close()

                if (videoBytes == null) {
                    Log.e("VideoUpload", "Cannot read video file")
                    throw Exception("Cannot read video file")
                }

                // Check file size (limit is 500MB)
                val fileSizeMB = videoBytes.size / (1024 * 1024)
                Log.d("VideoUpload", "Video file size: ${fileSizeMB}MB")

                if (fileSizeMB > 500) {
                    Log.e("VideoUpload", "Video size too large: ${fileSizeMB}MB")
                    throw Exception("Video size too large (${fileSizeMB}MB)")
                }

                // try uploading with multipart/form-data
                try {
                    Log.d("VideoUpload", "Multipart upload")
                    uploadVideoAsMultipart(videoBytes)
                } catch (e: Exception) {
                    Log.w("VideoUpload", "Multipart uploading failed, trying Base64 upload", e)

                    if (fileSizeMB > 100) {
                        Log.e("VideoUpload", "This file is too big, cannot use Base64: ${fileSizeMB}MB")
                        throw Exception("File size too big (${fileSizeMB}MB)，cannot use Base64 uploading")
                    }

                    uploadVideoAsBase64(videoBytes)
                }
                Log.i("VideoUpload", "Video upload successful")

            } catch (e: Exception) {
                Log.e("VideoUpload", "Video upload failed.", e)
                withContext(Dispatchers.Main) {
                    _uiState.update {
                        it.copy(
                            isLoading = false,
                            errorMessage = "Video upload failed: ${e.message}"
                        )
                    }
                }
            }
        }
    }

    // Multipart uploading
    private suspend fun uploadVideoAsMultipart(videoBytes: ByteArray) {
        Log.d("VideoUpload", "=== Multipart uploading start ===")

        try {
            val tempFile = File.createTempFile("upload_video", ".mp4", context.cacheDir)
            Log.d("VideoUpload", "Create temp file: ${tempFile.absolutePath}")

            tempFile.writeBytes(videoBytes)
            Log.d("VideoUpload", "Writing to temp file complete, size: ${tempFile.length()} bytes")

            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "video",
                    "video.mp4",
                    tempFile.asRequestBody("video/mp4".toMediaType())
                )
                .build()

            val url = "http://$serverIP:$serverPort/send-video"
            Log.d("VideoUpload", "Request URL: $url")

            val request = Request.Builder()
                .url(url)
                .post(requestBody)
                .build()

            Log.d("VideoUpload", "Sending HTTP request...")
            val response = httpClient.newCall(request).execute()

            // clear temp file
            val deleted = tempFile.delete()
            Log.d("VideoUpload", "Temp file deletion ${if (deleted) "successful" else "failed"}")

            Log.d("VideoUpload", "Received response, code: ${response.code}")

            if (response.isSuccessful) {
                val responseBody = response.body?.string()
                Log.d("VideoUpload", "Response: $responseBody")
                responseBody?.let { handleVideoResponse(it) }
            } else {
                val errorBody = response.body?.string()
                Log.e("VideoUpload", "Server error: $errorBody")
                throw Exception("Server error: ${response.code} - ${response.message}")
            }
        } catch (e: Exception) {
            Log.e("VideoUpload", "Multipart uploading failed", e)
            throw e
        }
    }

    // Base64 uploading
    private suspend fun uploadVideoAsBase64(videoBytes: ByteArray) {
        Log.d("VideoUpload", "=== Base64 uploading start ===")

        try {
            Log.d("VideoUpload", "Starting Base64 encoding...")
            val base64String = Base64.encodeToString(videoBytes, Base64.NO_WRAP)
            Log.d("VideoUpload", "Base64 encoded，length: ${base64String.length}")

            val jsonBody = JSONObject().apply {
                put("video", base64String)
            }

            val requestBody = jsonBody.toString().toRequestBody("application/json".toMediaType())
            val url = "http://$serverIP:$serverPort/send-video"
            Log.d("VideoUpload", "Request URL: $url")

            val request = Request.Builder()
                .url(url)
                .post(requestBody)
                .build()

            Log.d("VideoUpload", "Sending Base64 request...")
            val response = httpClient.newCall(request).execute()
            Log.d("VideoUpload", "Received response, code: ${response.code}")

            if (response.isSuccessful) {
                val responseBody = response.body?.string()
                Log.d("VideoUpload", "Response: $responseBody")
                responseBody?.let { handleVideoResponse(it) }
            } else {
                val errorBody = response.body?.string()
                Log.e("VideoUpload", "Server error: $errorBody")
                throw Exception("Server error: ${response.code} - ${response.message}")
            }
        } catch (e: Exception) {
            Log.e("VideoUpload", "Base64 uploading failed", e)
            throw e
        }
    }

    // Handle server response
    private suspend fun handleVideoResponse(responseBody: String) {
        Log.d("VideoUpload", "=== Handle Video Response ===")

        try {
            Log.d("VideoUpload", "Analyzing JSON response...")
            val jsonResponse = JSONObject(responseBody)
            val videoUrl = jsonResponse.optString("Video")
            val width = jsonResponse.optInt("Width", 0)
            val height = jsonResponse.optInt("Height", 0)

            Log.d("VideoUpload", "Result - URL: $videoUrl, size: ${width}x${height}")

            if (videoUrl.isNotEmpty()) {
                val videoUri = Uri.parse(videoUrl)
                Log.d("VideoUpload", "Parsed URI: $videoUri")

                withContext(Dispatchers.Main) {
                    _uiState.update {
                        it.copy(
                            isLoading = false,
                            processedVideoUri = videoUri,
                            selectedVideoUri = null
                        )
                    }
                }
                Log.i("VideoUpload", "Video processing complete，URI updated")
            } else {
                Log.e("VideoUpload", "No video URL present")
                throw Exception("No video URL present")
            }
        } catch (e: Exception) {
            Log.e("VideoUpload", "Server response handling failed", e)
            throw Exception("Server response handling failed: ${e.message}")
        }
    }

    fun fetchDemoData() {
        _uiState.update { it.copy(isLoading = true, errorMessage = null) }

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val jsonBody = JSONObject().apply {
                    put("title", "demo video")
                    put("videouri", "this is a test")
                }

                val requestBody = jsonBody.toString().toRequestBody("application/json".toMediaType())

                val request = Request.Builder()
                    .url("http://$serverIP:$serverPort/api/upload/")
                    .post(requestBody)
                    .build()

                val response = httpClient.newCall(request).execute()

                withContext(Dispatchers.Main) {
                    if (response.isSuccessful) {
                        _uiState.update {
                            it.copy(
                                isLoading = false,
                                errorMessage = null
                            )
                        }
                    } else {
                        _uiState.update {
                            it.copy(
                                isLoading = false,
                                errorMessage = "Fetch demo failed: ${response.code}"
                            )
                        }
                    }
                }

            } catch (e: Exception) {
                Log.e("DemoData", "Fetch demo failed", e)
                withContext(Dispatchers.Main) {
                    _uiState.update {
                        it.copy(
                            isLoading = false,
                            errorMessage = "Network error: ${e.message}"
                        )
                    }
                }
            }
        }
    }


    fun resetApp() {
        closeCamera()
        _uiState.update {
            CameraUiState(ipAddress = _uiState.value.ipAddress)
        }
    }


    fun clearError() {
        _uiState.update { it.copy(errorMessage = null) }
    }

    override fun onCleared() {
        super.onCleared()
        disconnectWebSocket()
        httpClient.dispatcher.executorService.shutdown()
    }
}