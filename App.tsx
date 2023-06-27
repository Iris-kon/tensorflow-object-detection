import * as cocoSsd from '@tensorflow-models/coco-ssd'
import * as tf from '@tensorflow/tfjs'
import { cameraWithTensors } from '@tensorflow/tfjs-react-native'
import { Camera, CameraType } from 'expo-camera'
import { StatusBar } from 'expo-status-bar'
import { useEffect, useRef, useState } from 'react'
import { Dimensions, LogBox, Platform, StyleSheet, View } from 'react-native'
import Canvas from 'react-native-canvas'

const TensorCamera = cameraWithTensors(Camera)

const { width, height } = Dimensions.get('window')

LogBox.ignoreAllLogs(true)
export default function App() {
  const [model, setModel] = useState<cocoSsd.ObjectDetection>()
  const [permission, requestPermission] = Camera.useCameraPermissions()

  let context = useRef<CanvasRenderingContext2D>()
  let canvas = useRef<Canvas>()

  function handleCameraStream(images: any) {
    console.log(images)
    const loop = async () => {
      const nextImageTensor = images.next().value

      if (!model || !nextImageTensor) {
        throw new Error('No model or image tensor')
      }

      model
        .detect(nextImageTensor)
        .then((prediction) => {
          drawRectangle(prediction, nextImageTensor)
        })
        .catch((e) => console.log(e))

      requestAnimationFrame(loop)
    }

    loop()
  }

  function drawRectangle(
    predictions: cocoSsd.DetectedObject[],
    nextImageTensor: any
  ) {
    if (!context.current || !canvas.current) {
      return
    }

    const scaleWidth = width / nextImageTensor.shape[1]
    const scaleHeight = height / nextImageTensor.shape[0]

    const flipHorizontal = Platform.OS === 'ios' ? false : true

    context.current.clearRect(0, 0, width, height)

    for (const prediction of predictions) {
      const [x, y, width, height] = prediction.bbox

      const boudingBoxX = flipHorizontal
        ? canvas.current.width - x * scaleWidth - width * scaleWidth
        : x * scaleWidth
      const boundingBoxY = y * scaleHeight

      context.current.strokeRect(
        boudingBoxX,
        boundingBoxY,
        width * scaleWidth,
        height * scaleHeight
      )

      context.current.strokeText(
        prediction.class,
        boudingBoxX - 5,
        boundingBoxY - 5
      )
    }
  }

  async function handleCanvas(can: Canvas) {
    can.width = width
    can.height = height

    const ctx: CanvasRenderingContext2D = can.getContext('2d')

    ctx.strokeStyle = 'red'
    ctx.fillStyle = 'red'
    ctx.lineWidth = 3

    context.current = ctx
    canvas.current = can
  }

  useEffect(() => {
    ;(async () => {
      requestPermission()
      await tf.ready()
      setModel(await cocoSsd.load())
    })()
  }, [])

  if (!permission?.granted) {
    return
  }
  return (
    <View style={styles.container}>
      <TensorCamera
        useCustomShadersToResize={false}
        style={styles.camera}
        type={CameraType.back}
        cameraTextureHeight={1080}
        cameraTextureWidth={1920}
        resizeHeight={200}
        resizeWidth={152}
        resizeDepth={3}
        onReady={handleCameraStream}
        autorender={true}
      />
      <Canvas style={styles.canvas} ref={handleCanvas} />
      <StatusBar style="auto" />
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  camera: {
    width: '100%',
    height: '100%',
  },
  canvas: {
    position: 'absolute',
    zIndex: 99999,
    width: '100%',
    height: '100%',
  },
})
