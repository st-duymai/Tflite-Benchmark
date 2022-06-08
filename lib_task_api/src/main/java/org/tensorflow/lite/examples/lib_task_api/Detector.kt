package org.tensorflow.lite.examples.lib_task_api

import android.graphics.Bitmap

interface Detector {

    fun benchmark(bitmap: Bitmap): Long
}
