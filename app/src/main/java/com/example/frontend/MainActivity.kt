package com.example.frontend

import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.frontend.utils.*
import com.example.frontend.viewmodel.CameraUiState
import com.example.frontend.viewmodel.CameraViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    // Camera Components
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService

    // Request permission
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (allGranted) {
            // Access granted. Further implementation is coded in ViewModel
        } else {
            showToast("Camera and storage permissions is required")
        }
    }

    // Select video
    private val selectVideoLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent() // Standard file/document selector
    ) { uri: Uri? ->
        uri?.let { viewModel.selectVideo(it) } // Handled in viewModel
    }

    private lateinit var viewModel: CameraViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()

        setContent {
            CameraAppTheme {
                viewModel = viewModel()
                MainScreen(
                    viewModel = viewModel,
                    onRequestPermissions = { requestPermissions() },
                    // "video/*" will only show video files (in file selector)
                    onSelectVideo = { selectVideoLauncher.launch("video/*") }
                )
            }
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun MainScreen(
        viewModel: CameraViewModel,
        onRequestPermissions: () -> Unit,
        onSelectVideo: () -> Unit
    ) {
        val context = LocalContext.current
        val uiState by viewModel.uiState.collectAsState()

        LaunchedEffect(uiState.errorMessage) {
            uiState.errorMessage?.let { message ->
                Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                viewModel.clearError()
            }
        }

        Box(modifier = Modifier.fillMaxSize()) {
            // Main page
            if (!uiState.isCameraOpen) {
                MainContent(
                    uiState = uiState,
                    onOpenCamera = {
                        if (PermissionUtils.hasAllPermissions(context)) {
                            viewModel.openCamera()
                        } else {
                            onRequestPermissions()
                        }
                    },
                    onSelectVideo = onSelectVideo,
                    onFetchDemo = { viewModel.fetchDemoData() },
                    onReset = { viewModel.resetApp() },
                    onSendVideo = { viewModel.sendVideoToServer() },
                    onSaveVideo = { uri ->
                        lifecycleScope.launch {
                            val success = MediaUtils.saveVideoToGallery(context, uri)
                            Toast.makeText(
                                context,
                                if (success) Constants.SUCCESS_VIDEO_SAVED else "Failed to Download",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    }
                )
            }

            // Camera Page
            if (uiState.isCameraOpen) {
                CameraScreen(
                    uiState = uiState,
                    onCloseCamera = { viewModel.closeCamera() },
                    onToggleRecording = { viewModel.toggleRecording() },
                    onImageCaptured = { imageBytes ->
                        viewModel.sendImageToServer(imageBytes)
                    }
                )
            }
        }
    }

    @Composable
    fun MainContent(
        uiState: CameraUiState,
        onOpenCamera: () -> Unit,
        onSelectVideo: () -> Unit,
        onFetchDemo: () -> Unit,
        onReset: () -> Unit,
        onSendVideo: () -> Unit,
        onSaveVideo: (Uri) -> Unit
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(32.dp))

            // Title
            Text(
                text = "Evaluator",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )

            Spacer(modifier = Modifier.height(16.dp))

            MainActionButton(
                text = "Choose Video",
                icon = Icons.Default.VideoLibrary,
                onClick = onSelectVideo,
                enabled = !uiState.isLoading
            )

            MainActionButton(
                text = "Open Camera",
                icon = Icons.Default.Camera,
                onClick = onOpenCamera,
                enabled = !uiState.isLoading
            )

            MainActionButton(
                text = "Fetch Data from API",
                icon = Icons.Default.CloudDownload,
                onClick = onFetchDemo,
                enabled = !uiState.isLoading,
                isLoading = uiState.isLoading
            )

            MainActionButton(
                text = "Back",
                icon = Icons.Default.Refresh,
                onClick = onReset,
                enabled = !uiState.isLoading
            )

            Spacer(modifier = Modifier.height(24.dp))

            // "Selected Video" block
            VideoProcessingSection(
                selectedVideoUri = uiState.selectedVideoUri,
                processedVideoUri = uiState.processedVideoUri,
                isLoading = uiState.isLoading,
                onSendVideo = onSendVideo,
                onSaveVideo = onSaveVideo
            )

            Spacer(modifier = Modifier.weight(1f))

            // IP address
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "IP Address",
                        fontSize = 14.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Text(
                        text = uiState.ipAddress.ifEmpty { "Fetching..." },
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
            }
        }
    }

    @Composable
    fun MainActionButton(
        text: String,
        icon: androidx.compose.ui.graphics.vector.ImageVector,
        onClick: () -> Unit,
        enabled: Boolean = true,
        isLoading: Boolean = false
    ) {
        Button(
            onClick = onClick,
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            shape = RoundedCornerShape(12.dp),
            enabled = enabled && !isLoading
        ) {
            if (isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.size(24.dp),
                    color = MaterialTheme.colorScheme.onPrimary,
                    strokeWidth = 3.dp
                )
                Spacer(modifier = Modifier.width(8.dp))
            } else {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
            }
            Text(
                text = text,
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium
            )
        }
    }

    @Composable
    fun VideoProcessingSection(
        selectedVideoUri: Uri?,
        processedVideoUri: Uri?,
        isLoading: Boolean,
        onSendVideo: () -> Unit,
        onSaveVideo: (Uri) -> Unit
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Video Selection",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface
                )

                Spacer(modifier = Modifier.height(12.dp))

                when {
                    selectedVideoUri != null -> {
                        Text(
                            text = "Video Selected",
                            fontSize = 14.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Spacer(modifier = Modifier.height(8.dp))

                        Button(
                            onClick = onSendVideo,
                            enabled = !isLoading,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            if (isLoading) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(20.dp),
                                    color = MaterialTheme.colorScheme.onPrimary,
                                    strokeWidth = 2.dp
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Loading...")
                            } else {
                                Icon(Icons.Default.Send, contentDescription = null)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Send Video")
                            }
                        }
                    }

                    processedVideoUri != null -> {
                        Text(
                            text = "video processing complete",
                            fontSize = 14.sp,
                            color = MaterialTheme.colorScheme.primary
                        )
                        Spacer(modifier = Modifier.height(8.dp))

                        Button(
                            onClick = { onSaveVideo(processedVideoUri) },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(Icons.Default.Download, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Save to Gallery")
                        }
                    }

                    isLoading -> {
                        CircularProgressIndicator(
                            modifier = Modifier.size(40.dp),
                            color = MaterialTheme.colorScheme.primary
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "Processing video...",
                            fontSize = 14.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    else -> {
                        Text(
                            text = "No video selected",
                            fontSize = 14.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
        }
    }

    @Composable
    fun CameraScreen(
        uiState: CameraUiState,
        onCloseCamera: () -> Unit,
        onToggleRecording: () -> Unit,
        onImageCaptured: (ByteArray) -> Unit
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
        ) {
            // Camera preview
            CameraPreview(
                modifier = Modifier.fillMaxSize(),
                onImageCaptured = onImageCaptured,
                isRecording = uiState.isRecording
            )

            // Overlay layer for displaying results
            DetectionOverlay(
                points = uiState.points,
                lines = uiState.lines,
                isRecording = uiState.isRecording,
                modifier = Modifier.fillMaxSize()
            )

            TopBar(
                onCloseCamera = onCloseCamera,
                modifier = Modifier.align(Alignment.TopCenter)
            )

            BottomBar(
                supination = uiState.supination,
                isRecording = uiState.isRecording,
                onToggleRecording = onToggleRecording,
                modifier = Modifier.align(Alignment.BottomCenter)
            )
        }
    }

    @Composable
    fun CameraPreview(
        modifier: Modifier = Modifier,
        onImageCaptured: (ByteArray) -> Unit,
        isRecording: Boolean
    ) {
        val lifecycleOwner = LocalLifecycleOwner.current
        val context = LocalContext.current
        val previewView = remember { PreviewView(context) }

        // temporary photo capturing logic - may change later
        LaunchedEffect(isRecording) {
            if (isRecording) {
                while (isRecording) {
                    delay(Constants.PHOTO_CAPTURE_INTERVAL_MS)
                    captureImage(onImageCaptured)
                }
            }
        }

        AndroidView(
            factory = { previewView },
            modifier = modifier
        ) {
            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                imageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview,
                        imageCapture
                    )
                } catch (exc: Exception) {
                    exc.printStackTrace()
                }
            }, ContextCompat.getMainExecutor(context))
        }
    }

    @Composable
    fun DetectionOverlay(
        points: List<com.example.frontend.viewmodel.Point>,
        lines: List<com.example.frontend.viewmodel.Line>,
        isRecording: Boolean,
        modifier: Modifier = Modifier
    ) {
        if (isRecording) {
            Canvas(modifier = modifier) {
                val scaleX = size.width / 4080f  // camera resolution?
                val scaleY = size.height / 3060f

                // draw points
                points.forEachIndexed { index, point ->
                    drawCircle(
                        color = getPointColor(index),
                        radius = Constants.POINT_RADIUS,
                        center = Offset(
                            point.x * scaleX,
                            point.y * scaleY
                        )
                    )
                }

                // draw lines
                lines.forEach { line ->
                    drawLine(
                        color = Color.Red,
                        start = Offset(
                            line.start.x * scaleX,
                            line.start.y * scaleY
                        ),
                        end = Offset(
                            line.end.x * scaleX,
                            line.end.y * scaleY
                        ),
                        strokeWidth = Constants.LINE_STROKE_WIDTH
                    )
                }
            }
        }
    }

    @Composable
    fun TopBar(
        onCloseCamera: () -> Unit,
        modifier: Modifier = Modifier
    ) {
        Card(
            modifier = modifier.padding(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = Color.Black.copy(alpha = 0.7f)
            )
        ) {
            Row(
                modifier = Modifier.padding(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onCloseCamera) {
                    Icon(
                        imageVector = Icons.Default.Close,
                        contentDescription = "Close Camera",
                        tint = Color.White
                    )
                }
                Text(
                    text = "Posture evaluation",
                    color = Color.White,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Medium
                )
            }
        }
    }

    @Composable
    fun BottomBar(
        supination: String,
        isRecording: Boolean,
        onToggleRecording: () -> Unit,
        modifier: Modifier = Modifier
    ) {
        Card(
            modifier = modifier.padding(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = Color.Black.copy(alpha = 0.7f)
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Posture info
                Text(
                    text = "Supinating: $supination",
                    color = Color.Red,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )

                Spacer(modifier = Modifier.height(16.dp))

                // "Recording" button
                Box(
                    modifier = Modifier
                        .size(80.dp)
                        .clip(CircleShape)
                        .background(
                            if (isRecording) Color.Red else Color.White
                        )
                        .clickable { onToggleRecording() },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = if (isRecording) Icons.Default.Stop else Icons.Default.RadioButtonUnchecked,
                        contentDescription = if (isRecording) "STOP" else "RECORD",
                        tint = if (isRecording) Color.White else Color.Black,
                        modifier = Modifier.size(32.dp)
                    )
                }
            }
        }
    }

    @Composable
    fun ConnectionIndicator(
        isConnected: Boolean,
        modifier: Modifier = Modifier
    ) {
        Card(
            modifier = modifier,
            colors = CardDefaults.cardColors(
                containerColor = if (isConnected) Color.Green else Color.Red
            )
        ) {
            Row(
                modifier = Modifier.padding(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = if (isConnected) Icons.Default.Wifi else Icons.Default.WifiOff,
                    contentDescription = null,
                    tint = Color.White,
                    modifier = Modifier.size(16.dp)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = if (isConnected) "connected" else "not connected",
                    color = Color.White,
                    fontSize = 12.sp
                )
            }
        }
    }

    // Helper functions
    private fun getPointColor(index: Int): Color {
        val red = (255 - index * 30).coerceAtLeast(0) / 255f
        val green = (index * 30).coerceAtMost(255) / 255f
        val blue = (255 - index * 30).coerceAtLeast(0) / 255f
        return Color(red, green, blue)
    }

    private fun captureImage(onImageCaptured: (ByteArray) -> Unit) {
        val imageCapture = imageCapture ?: return

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onError(exception: ImageCaptureException) {
                    exception.printStackTrace()
                }

                override fun onCaptureSuccess(image: ImageProxy) {
                    val imageBytes = CameraUtils.imageProxyToByteArray(image)
                    onImageCaptured(imageBytes)
                    image.close()
                }
            }
        )
    }

    private fun requestPermissions() {
        requestPermissionLauncher.launch(PermissionUtils.getRequiredPermissions())
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

// UI Theme
@Composable
fun CameraAppTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = Color(0xFF6200EE),
            onPrimary = Color.White,
            secondary = Color(0xFF03DAC6),
            onSecondary = Color.Black,
            surface = Color(0xFFF5F5F5),
            onSurface = Color(0xFF1C1B1F),
            background = Color.White,
            onBackground = Color(0xFF1C1B1F),
            error = Color(0xFFB00020),
            onError = Color.White
        ),
        content = content
    )
}