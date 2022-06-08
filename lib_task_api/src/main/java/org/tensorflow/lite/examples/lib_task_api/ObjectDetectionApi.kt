package org.tensorflow.lite.examples.lib_task_api

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions

class ObjectDetectionApi(private val objectDetector: ObjectDetector) : Detector {

    companion object {
        fun create(context: Context, modelFile: String): ObjectDetectionApi {
            val options = ObjectDetectorOptions.builder()
                .setBaseOptions(BaseOptions.builder().setNumThreads(1).useGpu().build())
                .setMaxResults(1)
                .build()
            val objectDetector = ObjectDetector.createFromFileAndOptions(
                context, modelFile, options
            )
            return ObjectDetectionApi(objectDetector)
        }
    }

    override fun benchmark(bitmap: Bitmap): Long {
        val startTime = System.nanoTime()
        objectDetector.detect(TensorImage.fromBitmap(bitmap))
        return (System.nanoTime() - startTime) / 1000000 // convert nanosecond to milisecond
    }
}
