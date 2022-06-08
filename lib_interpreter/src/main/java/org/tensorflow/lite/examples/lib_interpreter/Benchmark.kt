package org.tensorflow.lite.examples.lib_interpreter

import android.graphics.Bitmap

interface Benchmark {

    fun benchmark(bitmap: Bitmap): Long

    fun close()
}
