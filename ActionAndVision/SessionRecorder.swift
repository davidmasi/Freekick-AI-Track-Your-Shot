import AVFoundation

final class SessionRecorder {
    static let didFinishRecordingNotification = Notification.Name("SessionRecorderDidFinishRecording")
    static let didFailRecordingNotification = Notification.Name("SessionRecorderDidFailRecording")
    enum SessionRecorderError: Error {
        case alreadyRecording
        case writerNotReady
        case invalidSampleBuffer
        case finishFailed(Error?)
    }

    private let writingQueue = DispatchQueue(label: "com.freekick.sessionRecorder")
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var audioInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var startedSession = false
    private var frameCount = 0
    private var currentOutputURL: URL?
    private(set) var isRecording = false
    private(set) var prefersAudio = false
    private(set) var outputOrientation: AVCaptureVideoOrientation = .landscapeRight

    init(recordAudio: Bool = false) {
        self.prefersAudio = recordAudio
    }

    func startRecording(outputOrientation: AVCaptureVideoOrientation = .landscapeRight) throws {
        try writingQueue.sync {
            guard !isRecording else {
                throw SessionRecorderError.alreadyRecording
            }

            let outputURL = SessionStore.shared.makeTempSessionFileURL()
            currentOutputURL = outputURL
            self.outputOrientation = outputOrientation
            isRecording = true
            assetWriter = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
            startedSession = false
        }
    }

    func appendVideoSampleBuffer(_ sampleBuffer: CMSampleBuffer) {
        writingQueue.async { [weak self] in
            guard let self = self, self.isRecording else { return }
            do {
                try self.prepareVideoWriterIfNeeded(with: sampleBuffer)
                try self.appendVideoBuffer(sampleBuffer)
            } catch {
                // append failures are non-fatal; drop the frame
            }
        }
    }

    func appendAudioSampleBuffer(_ sampleBuffer: CMSampleBuffer) {
        writingQueue.async { [weak self] in
            guard let self = self, self.isRecording, self.prefersAudio else { return }
            do {
                try self.prepareAudioWriterIfNeeded(with: sampleBuffer)
                try self.appendAudioBuffer(sampleBuffer)
            } catch {
                // append failures are non-fatal; drop the frame
            }
        }
    }

    func finishRecording(completion: @escaping (Result<URL, Error>) -> Void) {
        writingQueue.async { [weak self] in
            guard let self = self else { return }
            guard let assetWriter = self.assetWriter, let outputURL = self.currentOutputURL else {
                completion(.failure(SessionRecorderError.writerNotReady))
                return
            }

            self.videoInput?.markAsFinished()
            self.audioInput?.markAsFinished()
            self.isRecording = false

            assetWriter.finishWriting { [weak self] in
                guard let self = self else { return }
                let localURL = outputURL

                self.resetWriter()

                if assetWriter.status == .completed {
                    DispatchQueue.main.async {
                        NotificationCenter.default.post(name: SessionRecorder.didFinishRecordingNotification, object: self, userInfo: ["url": localURL])
                        completion(.success(localURL))
                    }
                } else {
                    DispatchQueue.main.async {
                        NotificationCenter.default.post(name: SessionRecorder.didFailRecordingNotification, object: self, userInfo: ["error": assetWriter.error as Any])
                        completion(.failure(SessionRecorderError.finishFailed(assetWriter.error)))
                    }
                }
            }
        }
    }

    private func resetWriter() {
        self.assetWriter = nil
        self.videoInput = nil
        self.audioInput = nil
        self.pixelBufferAdaptor = nil
        self.startedSession = false
        self.frameCount = 0
        self.currentOutputURL = nil
    }

    private func prepareVideoWriterIfNeeded(with sampleBuffer: CMSampleBuffer) throws {
        if videoInput != nil {
            return
        }

        guard let assetWriter = assetWriter else {
            throw SessionRecorderError.writerNotReady
        }

        guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer) else {
            throw SessionRecorderError.invalidSampleBuffer
        }

        let dimensions = CMVideoFormatDescriptionGetPresentationDimensions(formatDescription, usePixelAspectRatio: true, useCleanAperture: true)

        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: Int(dimensions.width),
            AVVideoHeightKey: Int(dimensions.height)
        ]

        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
        videoInput.expectsMediaDataInRealTime = true
        videoInput.transform = self.transform(for: outputOrientation)

        let pixelFormatType = CMFormatDescriptionGetMediaSubType(formatDescription)
        let sourcePixelBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(pixelFormatType),
            kCVPixelBufferWidthKey as String: Int(dimensions.width),
            kCVPixelBufferHeightKey as String: Int(dimensions.height)
        ]

        guard assetWriter.canAdd(videoInput) else {
            throw SessionRecorderError.writerNotReady
        }

        assetWriter.add(videoInput)
        self.videoInput = videoInput
        self.pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoInput,
                                                                       sourcePixelBufferAttributes: sourcePixelBufferAttributes)

        if prefersAudio {
            try prepareAudioWriterIfNeeded(with: sampleBuffer)
        }

        assetWriter.startWriting()
    }

    private func prepareAudioWriterIfNeeded(with sampleBuffer: CMSampleBuffer) throws {
        if audioInput != nil {
            return
        }

        guard let assetWriter = assetWriter else {
            throw SessionRecorderError.writerNotReady
        }

        guard CMSampleBufferGetFormatDescription(sampleBuffer) != nil else {
            throw SessionRecorderError.invalidSampleBuffer
        }

        let audioInput = AVAssetWriterInput(mediaType: .audio, outputSettings: nil)
        audioInput.expectsMediaDataInRealTime = true

        guard assetWriter.canAdd(audioInput) else {
            throw SessionRecorderError.writerNotReady
        }

        assetWriter.add(audioInput)
        self.audioInput = audioInput
    }

    private func appendVideoBuffer(_ sampleBuffer: CMSampleBuffer) throws {
        guard let assetWriter = assetWriter,
              let videoInput = videoInput,
              let pixelBufferAdaptor = pixelBufferAdaptor,
              let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            throw SessionRecorderError.invalidSampleBuffer
        }

        let presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

        if !startedSession {
            assetWriter.startSession(atSourceTime: presentationTime)
            startedSession = true
        }

        guard videoInput.isReadyForMoreMediaData else {
            return
        }

        if pixelBufferAdaptor.append(imageBuffer, withPresentationTime: presentationTime) {
            frameCount += 1
        }
    }

    private func appendAudioBuffer(_ sampleBuffer: CMSampleBuffer) throws {
        guard let audioInput = audioInput else {
            return
        }

        guard audioInput.isReadyForMoreMediaData else {
            return
        }

        audioInput.append(sampleBuffer)
    }

    private func transform(for orientation: AVCaptureVideoOrientation) -> CGAffineTransform {
        switch orientation {
        case .portrait:
            return CGAffineTransform(rotationAngle: .pi / 2)
        case .portraitUpsideDown:
            return CGAffineTransform(rotationAngle: -.pi / 2)
        case .landscapeLeft:
            return CGAffineTransform(rotationAngle: .pi)
        case .landscapeRight:
            return .identity
        @unknown default:
            return .identity
        }
    }
}
