package org.tensorflow.lite.examples.androidbenchmark

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.work.*
import org.tensorflow.lite.examples.androidbenchmark.MainActivity.Companion.USE_GPU
import org.tensorflow.lite.examples.androidbenchmark.model.ModelType
import org.tensorflow.lite.examples.androidbenchmark.model.TfliteModel
import org.tensorflow.lite.examples.lib_interpreter.BenchmarkInterpreter
import org.tensorflow.lite.examples.lib_interpreter.InputDataType
import org.tensorflow.lite.examples.lib_taskapi.BenchMarkImageClassificationApi
import org.tensorflow.lite.examples.lib_taskapi.BenchMarkImageSegmenterApi
import org.tensorflow.lite.examples.lib_taskapi.BenchMarkObjectDetectionApi
import java.io.IOException
import java.io.InputStream
import java.lang.Exception

class MainActivity : AppCompatActivity() {

    companion object {
        const val NUMBER_OF_BENCHMARK = 50
        const val USE_GPU = true
        val models = listOf(
            TfliteModel(
                "efficientnet_lite0",
                "models/classification/efficient_net_lite0.tflite",
                224,
                224,
                InputDataType.UINT8,
                ModelType.CLASSIFICATION
            ),
            TfliteModel(
                "mnasnet",
                "models/classification/mnasnet.tflite",
                224,
                224,
                InputDataType.FLOAT32,
                ModelType.CLASSIFICATION
            ),
            TfliteModel(
                "mobilenet_v1_1",
                "models/classification/mobilenet_v1_quantized.tflite",
                224,
                224,
                InputDataType.UINT8,
                ModelType.CLASSIFICATION
            ),
//            TfliteModel(
//                "deeplabv3",
//                "models/image_segmenter/deeplabv3.tflite",
//                257,
//                257,
//                InputDataType.FLOAT32,
//                ModelType.SEGMENTATION
//            ),
//            TfliteModel(
//                "efficientdet_lite3_detection",
//                "models/object_detection/efficient_net_lite3_detection.tflite",
//                512,
//                512,
//                InputDataType.UINT8,
//                ModelType.DETECTION
//            ),
//            TfliteModel(
//                "ssd_mobilenet_v1",
//                "models/object_detection/ssd_mobilenet_v1.tflite",
//                300,
//                300,
//                InputDataType.UINT8,
//                ModelType.DETECTION
//            ),
            TfliteModel(
                "EfficientnetLite4fp32",
                "models/classification/efficientnet_lite4_fp32.tflite",
                300,
                300,
                InputDataType.FLOAT32,
                ModelType.CLASSIFICATION
            ),
            TfliteModel(
                "EfficientnetLite4int8",
                "models/classification/efficientnet_lite4_int8.tflite",
                300,
                300,
                InputDataType.UINT8,
                ModelType.CLASSIFICATION
            ),
            TfliteModel(
                "EfficientnetLite4uint8",
                "models/classification/efficientnet_lite4_uint8.tflite",
                300,
                300,
                InputDataType.UINT8,
                ModelType.CLASSIFICATION
            )
        )

        fun getBitmapFromAsset(context: Context, filePath: String?): Bitmap? {
            val assetManager: AssetManager = context.assets
            val istr: InputStream
            var bitmap: Bitmap? = null
            try {
                istr = assetManager.open(filePath!!)
                bitmap = BitmapFactory.decodeStream(istr)
            } catch (e: IOException) {
                // handle exception
            }
            return bitmap
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val btnBenchMarkInterpreter = findViewById<Button>(R.id.btnBenchMarkInterpreter)
        val workManager = WorkManager.getInstance(this)

        btnBenchMarkInterpreter.setOnClickListener {
            val benchmarkInterpreter =
                OneTimeWorkRequestBuilder<BenchmarkWorker>().build()
            workManager.getWorkInfoByIdLiveData(benchmarkInterpreter.id).observe(this) {
                handleWorkState(it.state)
            }
            workManager.enqueue(benchmarkInterpreter)
        }
    }

    private fun handleWorkState(state: WorkInfo.State) {
        if (state == WorkInfo.State.RUNNING) {
            Toast.makeText(this, "Benchmark is running.", Toast.LENGTH_LONG).show()
        } else if (state == WorkInfo.State.SUCCEEDED) {
            Toast.makeText(
                this,
                "Benchmark is done. Please check the logcat to see the results.",
                Toast.LENGTH_LONG
            ).show()
        } else if (state == WorkInfo.State.FAILED) {
            Toast.makeText(
                this,
                "Benchmark failed.",
                Toast.LENGTH_LONG
            ).show()
        }
    }
}

class BenchmarkWorker(appContext: Context, workerParams: WorkerParameters) :
    Worker(appContext, workerParams) {
    companion object {
        private const val TAG_1 = "Interpreter execute time:"
        private const val TAG_2 = "Task Api execute time:"
    }

    override fun doWork(): Result {
        return try {
            MainActivity.getBitmapFromAsset(applicationContext, "images/img_object.jpeg")
                ?.let { image ->
                    MainActivity.models.forEach { model ->
                        benchMarkInterpreter(model, image)
                        benchMarkTaskApi(model, image)
                    }
                    Log.d("Benchmark execute:", "Done")
                }
            Result.success()
        } catch (e: Exception) {
            Result.failure()
        }
    }

    private fun benchMarkInterpreter(model: TfliteModel, bitmap: Bitmap) {
        val benchMark = BenchmarkInterpreter.create(
            applicationContext, model.path,
            model.inputWidth,
            model.inputHeight,
            model.inputDataType,
            USE_GPU
        )
        val executeTimes = mutableListOf<Long>()
        (0 until MainActivity.NUMBER_OF_BENCHMARK).forEach { _ ->
            executeTimes.add(benchMark.benchmark(bitmap))
        }
        Log.d("$TAG_1 ${model.name}:", "${executeTimes.average().toInt()}ms")
        benchMark.close()
    }

    private fun benchMarkTaskApi(model: TfliteModel, bitmap: Bitmap) {
        val benchMark = when (model.modelType) {
            ModelType.CLASSIFICATION -> BenchMarkImageClassificationApi.create(
                applicationContext,
                model.path,
                USE_GPU
            )
            ModelType.SEGMENTATION -> BenchMarkImageSegmenterApi.create(
                applicationContext,
                model.path,
                USE_GPU
            )
            else -> BenchMarkObjectDetectionApi.create(applicationContext, model.path, USE_GPU)
        }
        val executeTimes = mutableListOf<Long>()
        (0 until MainActivity.NUMBER_OF_BENCHMARK).forEach { _ ->
            executeTimes.add(benchMark.benchmark(bitmap))
        }
        Log.d("$TAG_2 ${model.name}:", "${executeTimes.average().toInt()}ms")
        benchMark.close()
    }
}
