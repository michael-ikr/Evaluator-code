package com.example.frontend.utils

import java.util.regex.Pattern

object ValidationUtils {

    private val IP_ADDRESS_PATTERN = Pattern.compile(
        "^(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.){3}([01]?\\d\\d?|2[0-4]\\d|25[0-5])$"
    )

    private val EMAIL_PATTERN = Pattern.compile(
        "^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$"
    )

    fun isValidIpAddress(ip: String?): Boolean {
        if (ip.isNullOrBlank()) return false
        return IP_ADDRESS_PATTERN.matcher(ip.trim()).matches()
    }


    fun isValidPort(port: Int): Boolean {
        return port in 1..65535
    }

    fun isValidPort(port: String?): Boolean {
        if (port.isNullOrBlank()) return false
        return try {
            val portInt = port.trim().toInt()
            isValidPort(portInt)
        } catch (e: NumberFormatException) {
            false
        }
    }

    fun hasValidExtension(fileName: String?, validExtensions: List<String>): Boolean {
        if (fileName.isNullOrBlank()) return false

        val extension = fileName.substringAfterLast('.', "").lowercase()
        return validExtensions.any { it.lowercase() == extension }
    }

    fun isValidVideoExtension(fileName: String?): Boolean {
        val videoExtensions = listOf("mp4", "avi", "mov", "wmv", "flv", "webm", "mkv")
        return hasValidExtension(fileName, videoExtensions)
    }

    fun isValidImageExtension(fileName: String?): Boolean {
        val imageExtensions = listOf("jpg", "jpeg", "png", "gif", "bmp", "webp")
        return hasValidExtension(fileName, imageExtensions)
    }
}