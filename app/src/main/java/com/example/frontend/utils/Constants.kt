package com.example.frontend.utils

object Constants {
    // Network Configuration
    const val DEFAULT_SERVER_IP = "10.186.30.166"
    const val DEFAULT_SERVER_PORT = 8000
    const val WEBSOCKET_ENDPOINT = "/ws"
    const val UPLOAD_ENDPOINT = "/send-video"
    const val DEMO_ENDPOINT = "/api/upload/"

    // Camera Configuration
    const val PHOTO_CAPTURE_INTERVAL_MS = 500L
    const val IMAGE_CAPTURE_QUALITY = 80

    // UI Settings
    const val POINT_RADIUS = 20f
    const val LINE_STROKE_WIDTH = 5f

    // WebSocket Message Type
    const val WS_MESSAGE_TYPE_FRAME = "frame"
    const val WS_MESSAGE_TYPE_CONTROL = "control"

    // Error Messages
    const val ERROR_CAMERA_PERMISSION = "Camera Permission Required"
    const val ERROR_STORAGE_PERMISSION = "Storage Permission Required"
    const val ERROR_NETWORK_UNAVAILABLE = "Network Unavailable"
    const val ERROR_SERVER_UNREACHABLE = "Server Unreachable"
    const val ERROR_FILE_TOO_LARGE = "File too large, Please select a smaller file"
    const val ERROR_INVALID_FILE = "Invalid File"

    // Success Messages
    const val SUCCESS_VIDEO_SAVED = "Video Saved"
    const val SUCCESS_IMAGE_SENT = "Image Sent"
    const val SUCCESS_CONNECTION_ESTABLISHED = "Connection Established"
}