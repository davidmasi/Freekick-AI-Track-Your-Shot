/*
See LICENSE folder for this sample's licensing information.

Abstract:
View controller responsible for the setup state of the game.
     The setup consists of the following tasks:
     - goal detection
     - goal placement check
     - goal contours analysis
     - scene stability check
*/

import UIKit
import AVFoundation
import Vision

class SetupViewController: UIViewController {

    @IBOutlet var statusLabel: OverlayLabel!
    private struct GoalLineCandidate {
        let start: CGPoint
        let end: CGPoint
        let length: CGFloat
        let angle: CGFloat
        let midpoint: CGPoint
        let score: CGFloat
    }

    private enum GoalSide {
        case left
        case right
    }

    private let gameManager = GameManager.shared
    private let goalLocationGuide = BoundingBoxView()
    private let goalBoundingBox = BoundingBoxView()
    private let detectorDebugLabel = UILabel()

    private var selectedGoalSide: GoalSide?
    private var hasPresentedGoalSidePrompt = false
    private var goalDetectionRequest: VNCoreMLRequest!
    private let goalDetectionMinConfidence: VNConfidence = 0.2 // was 0.6

    enum SceneSetupStage {
        case detectingGoal
        case detectingGoalPlacement
        case detectingSceneStability
        case detectingGoalContours
        case setupComplete
    }

    private var setupStage = SceneSetupStage.detectingGoal

    enum SceneStabilityResult {
        case unknown
        case stable
        case unstable
    }

    private let sceneStabilityRequestHandler = VNSequenceRequestHandler()
    private let sceneStabilityRequiredHistoryLength = 15
    private var sceneStabilityHistoryPoints = [CGPoint]()
    private var previousSampleBuffer: CMSampleBuffer?

    override func viewDidLoad() {
        super.viewDidLoad()
        goalLocationGuide.borderColor = #colorLiteral(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
        goalLocationGuide.borderWidth = 3
        goalLocationGuide.borderCornerRadius = 4
        goalLocationGuide.borderCornerSize = 30
        goalLocationGuide.backgroundOpacity = 0.25
        goalLocationGuide.isHidden = true
        view.addSubview(goalLocationGuide)
        goalBoundingBox.borderColor = #colorLiteral(red: 1, green: 0.5763723254, blue: 0, alpha: 1)
        goalBoundingBox.borderWidth = 2
        goalBoundingBox.borderCornerRadius = 4
        goalBoundingBox.borderCornerSize = 0
        goalBoundingBox.backgroundOpacity = 0.45
        goalBoundingBox.isHidden = true
        view.addSubview(goalBoundingBox)

        detectorDebugLabel.translatesAutoresizingMaskIntoConstraints = false
        detectorDebugLabel.numberOfLines = 0
        detectorDebugLabel.font = UIFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        detectorDebugLabel.textColor = .white
        detectorDebugLabel.backgroundColor = UIColor.black.withAlphaComponent(0.45)
        detectorDebugLabel.textAlignment = .left
        detectorDebugLabel.layer.cornerRadius = 4
        detectorDebugLabel.layer.masksToBounds = true
        detectorDebugLabel.text = " Detector: idle "
        view.addSubview(detectorDebugLabel)
        NSLayoutConstraint.activate([
            detectorDebugLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            detectorDebugLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -12),
            detectorDebugLabel.widthAnchor.constraint(lessThanOrEqualToConstant: 220)
        ])

        updateSetupState()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        do {
            // Create Vision request based on CoreML model
            let model = try VNCoreMLModel(for: GoalDetector(configuration: MLModelConfiguration()).model)
            goalDetectionRequest = VNCoreMLRequest(model: model)
            // Since goal is close to the side of a landscape image,
            // we need to set crop and scale option to scaleFit.
            // By default vision request will run on centerCrop.
            goalDetectionRequest.imageCropAndScaleOption = .scaleFit
        } catch {
            let error = AppError.createRequestError(reason: "Could not create Vision request for goal detector")
            AppError.display(error, inViewController: self)
        }

        if gameManager.recordedVideoSource == nil && !hasPresentedGoalSidePrompt {
            hasPresentedGoalSidePrompt = true
            presentGoalSideSelectionPrompt()
        }
    }

    func updateBoundingBox(_ boundingBox: BoundingBoxView, withViewRect rect: CGRect?, visionRect: CGRect) {
        DispatchQueue.main.async {
            boundingBox.frame = rect ?? .zero
            boundingBox.visionRect = visionRect
            if rect == nil {
                boundingBox.perform(transition: .fadeOut, duration: 0.1)
            } else {
                boundingBox.perform(transition: .fadeIn, duration: 0.1)
            }
        }
    }

    func updateSetupState() {
        let goalBox = goalBoundingBox
        DispatchQueue.main.async {
            switch self.setupStage {
            case .detectingGoal:
                if self.gameManager.recordedVideoSource == nil && self.selectedGoalSide == nil {
                    self.statusLabel.text = "Select Goal Placement"
                } else {
                    self.statusLabel.text = "Locating Goal"
                }
                self.statusLabel.textColor = #colorLiteral(red: 0.501960814, green: 0.501960814, blue: 0.501960814, alpha: 1)
            case .detectingGoalPlacement:
                // Goal placement guide is shown only when using camera feed.
                // Otherwise we always assume the goal is placed correctly.
                var boxPlacedCorrectly = true
                if !self.goalLocationGuide.isHidden {
                    boxPlacedCorrectly = goalBox.containedInside(self.goalLocationGuide)
                }
                goalBox.borderColor = boxPlacedCorrectly ? #colorLiteral(red: 0.4641711116, green: 1, blue: 0, alpha: 1) : #colorLiteral(red: 1, green: 0.5763723254, blue: 0, alpha: 1)
                if boxPlacedCorrectly {
                    self.statusLabel.text = "Keep Device Stationary"
                    self.statusLabel.textColor = #colorLiteral(red: 0.501960814, green: 0.501960814, blue: 0.501960814, alpha: 1)
                    self.setupStage = .detectingSceneStability
                } else {
                    self.statusLabel.text = "Place Goal into the Box"
                    self.statusLabel.textColor = #colorLiteral(red: 0.501960814, green: 0.501960814, blue: 0.501960814, alpha: 1)
                }
            case .detectingSceneStability:
                switch self.sceneStability {
                case .unknown:
                    break
                case .unstable:
                    self.previousSampleBuffer = nil
                    self.sceneStabilityHistoryPoints.removeAll()
                    self.setupStage = .detectingGoalPlacement
                case .stable:
                    self.setupStage = .detectingGoalContours
                }
            default:
                break
            }
        }
    }

    private var guideVisionRectForSelectedSide: CGRect? {
        guard let side = selectedGoalSide else { return nil }
        switch side {
        case .right:
            return CGRect(x: 0.7, y: 0.3, width: 0.28, height: 0.3)
        case .left:
            return CGRect(x: 0.02, y: 0.3, width: 0.28, height: 0.3)
        }
    }

    private func presentGoalSideSelectionPrompt() {
        let alert = UIAlertController(
            title: "Goal Side",
            message: "Select whether the goal will be located on the left or right side of the view.",
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "Left", style: .default, handler: { _ in
            self.selectedGoalSide = .left
            self.setupStage = .detectingGoal
            self.updateSetupState()
        }))
        alert.addAction(UIAlertAction(title: "Right", style: .default, handler: { _ in
            self.selectedGoalSide = .right
            self.setupStage = .detectingGoal
            self.updateSetupState()
        }))
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: { _ in
            (self.parent as? RootViewController)?.exitToMenu()
        }))
        present(alert, animated: true)
    }

    func analyzeGoalContours(_ contours: [VNContour]) -> CGPath? {
        let simplifiedContours = contours.compactMap { contour -> VNContour? in
            guard let poly = try? contour.polygonApproximation(epsilon: 0.01),
                  poly.pointCount >= 3 else {
                return nil
            }
            return poly
        }

        let lineCandidates = extractGoalLineCandidates(from: simplifiedContours)

        guard let topBar = selectBestTopBar(from: lineCandidates),
              let leftPost = selectBestPost(from: lineCandidates, side: .left, topBar: topBar),
              let rightPost = selectBestPost(from: lineCandidates, side: .right, topBar: topBar)
        else {
            return nil
        }

        let topLeft = intersection(topBar, leftPost) ?? topBar.start
        let topRight = intersection(topBar, rightPost) ?? topBar.end
        let bottomLeft = lowerEndpoint(of: leftPost)
        let bottomRight = lowerEndpoint(of: rightPost)

        let path = UIBezierPath()
        path.move(to: topLeft)
        path.addLine(to: topRight)
        path.addLine(to: bottomRight)
        path.addLine(to: bottomLeft)
        path.close()

        return path.cgPath
    }

    // Compute Bounding Box Area
    func boundingBoxArea(_ contour: VNContour) -> CGFloat {
        let points = contour.normalizedPoints.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
        guard let minX = points.map({ $0.x }).min(),
              let maxX = points.map({ $0.x }).max(),
              let minY = points.map({ $0.y }).min(),
              let maxY = points.map({ $0.y }).max() else {
            return 0
        }
        return (maxX - minX) * (maxY - minY)
    }

    var sceneStability: SceneStabilityResult {
        // Determine if we have enough evidence of stability.
        guard sceneStabilityHistoryPoints.count > sceneStabilityRequiredHistoryLength else {
            return .unknown
        }

        // Calculate the moving average by adding up values of stored points
        // returned by VNTranslationalImageRegistrationRequest for both axis
        var movingAverage = CGPoint.zero
        movingAverage.x = sceneStabilityHistoryPoints.map { $0.x }.reduce(.zero, +)
        movingAverage.y = sceneStabilityHistoryPoints.map { $0.y }.reduce(.zero, +)
        // Get the moving distance by adding absolute moving average values of individual axis
        let distance = abs(movingAverage.x) + abs(movingAverage.y)
        // If the distance is not significant enough to affect the game analysis (less that 10 points),
        // we declare the scene being stable
        return (distance < 10 ? .stable : .unstable)
    }
}

extension SetupViewController: CameraViewControllerOutputDelegate {
    func cameraViewController(_ controller: CameraViewController, didReceiveBuffer buffer: CMSampleBuffer, orientation: CGImagePropertyOrientation) {
        do {
            switch setupStage {
            case .setupComplete:
                // Setup is complete - no reason to run vision requests.
                return
            case .detectingSceneStability:
                try checkSceneStability(controller, buffer, orientation)
            case .detectingGoalContours:
                try detectGoalContours(controller, buffer, orientation)
            case .detectingGoal, .detectingGoalPlacement:
                guard gameManager.recordedVideoSource != nil || selectedGoalSide != nil else {
                    return
                }
                try detectGoal(controller, buffer, orientation)
            }
            updateSetupState()
        } catch {
            AppError.display(error, inViewController: self)
        }
    }

    private func checkSceneStability(_ controller: CameraViewController, _ buffer: CMSampleBuffer, _ orientation: CGImagePropertyOrientation) throws {
        guard let previousBuffer = self.previousSampleBuffer else {
            self.previousSampleBuffer = buffer
            return
        }
        let registrationRequest = VNTranslationalImageRegistrationRequest(targetedCMSampleBuffer: buffer)
        try sceneStabilityRequestHandler.perform([registrationRequest], on: previousBuffer, orientation: orientation)
        self.previousSampleBuffer = buffer
        if let alignmentObservation = registrationRequest.results?.first as? VNImageTranslationAlignmentObservation {
            let transform = alignmentObservation.alignmentTransform
            sceneStabilityHistoryPoints.append(CGPoint(x: transform.tx, y: transform.ty))
        }
    }

    fileprivate func detectGoal(_ controller: CameraViewController, _ buffer: CMSampleBuffer, _ orientation: CGImagePropertyOrientation) throws {
        // This is where we detect the goal.
        let visionHandler = VNImageRequestHandler(cmSampleBuffer: buffer, orientation: orientation, options: [:])
        try visionHandler.perform([goalDetectionRequest])
        var rect: CGRect?
        var visionRect = CGRect.null
        var rawResults: [VNDetectedObjectObservation] = []
        var chosenConfidence: VNConfidence = 0
        if let results = goalDetectionRequest.results as? [VNDetectedObjectObservation] {
            rawResults = results
            // Filter out classification results with low confidence
            let filteredResults = results.filter { $0.confidence > goalDetectionMinConfidence }
            // If a goal side has been selected, prefer detections on that side.
            let sideFilteredResults = filteredResults.filter { observation in
                guard let selectedSide = self.selectedGoalSide else {
                    return true
                }
                let midX = observation.boundingBox.midX
                switch selectedSide {
                case .left:
                    return midX <= 0.5
                case .right:
                    return midX >= 0.5
                }
            }
            let chosenResults = sideFilteredResults.isEmpty ? filteredResults : sideFilteredResults
            if !chosenResults.isEmpty {
                visionRect = chosenResults[0].boundingBox
                rect = controller.viewRectForVisionRect(visionRect)
                chosenConfidence = chosenResults[0].confidence
            }
        }
        updateDetectorDebugLabel(rawResults: rawResults, chosenConfidence: chosenConfidence)
        // Show goal placement guide only when using camera feed.
        if gameManager.recordedVideoSource == nil {
            if let guideVisionRect = guideVisionRectForSelectedSide {
                let guideRect = controller.viewRectForVisionRect(guideVisionRect)
                updateBoundingBox(goalLocationGuide, withViewRect: guideRect, visionRect: guideVisionRect)
            } else {
                DispatchQueue.main.async {
                    self.goalLocationGuide.isHidden = true
                }
            }
        }
        updateBoundingBox(goalBoundingBox, withViewRect: rect, visionRect: visionRect)
        // If rect is nil we need to keep looking for the board, otherwise check the goal placement
        self.setupStage = (rect == nil) ? .detectingGoal : .detectingGoalPlacement
    }

    private func detectGoalContours(_ controller: CameraViewController, _ buffer: CMSampleBuffer, _ orientation: CGImagePropertyOrientation) throws {
        let visionHandler = VNImageRequestHandler(cmSampleBuffer: buffer, orientation: orientation, options: [:])
        let contoursRequest = VNDetectContoursRequest()
        contoursRequest.contrastAdjustment = 1.6 // Adjust contrast for better results
        let roi = clampedToUnitRect(goalBoundingBox.visionRect)
        guard roi.width > 0, roi.height > 0 else { return }
        contoursRequest.regionOfInterest = roi

        try visionHandler.perform([contoursRequest])

        if let result = contoursRequest.results?.first as? VNContoursObservation {
            // Analyze detected contours
            guard let goalFramePath = analyzeGoalContours(result.topLevelContours) else {
                return
            }

            DispatchQueue.main.async {
                self.gameManager.goalRegion = self.goalBoundingBox.frame
                // Convert pixels to meters using the ML-detected goal width (stable frame to frame,
                // unlike the contour bounding box). 7.32m is the regulation distance between posts.
                self.gameManager.pointToMeterMultiplier = GameConstants.goalLength / Double(self.goalBoundingBox.frame.width)

                if let imageBuffer = CMSampleBufferGetImageBuffer(buffer) {
                    // Render the CIImage to a real CGImage right now. Don't store a CIImage-backed
                    // UIImage — those defer GPU rendering until first use, which can crash later when
                    // jpegData() forces the pipeline against a recycled/invalid camera buffer.
                    let ciImage = CIImage(cvImageBuffer: imageBuffer).oriented(orientation)
                    let context = CIContext(options: nil)
                    if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
                        self.gameManager.previewImage = UIImage(cgImage: cgImage)
                    }
                }

                self.goalBoundingBox.visionPath = goalFramePath
                self.goalBoundingBox.borderColor = #colorLiteral(red: 1, green: 1, blue: 1, alpha: 0.199807363)
                self.gameManager.stateMachine.enter(GameManager.DetectedGoalState.self)
            }
        }
    }

    private func extractGoalLineCandidates(from contours: [VNContour]) -> [GoalLineCandidate] {
        var candidates: [GoalLineCandidate] = []

        for contour in contours {
            let points = contour.normalizedPoints.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
            guard points.count >= 2 else { continue }

            for (p1, p2) in zip(points, points.dropFirst()) {
                let dx = p2.x - p1.x
                let dy = p2.y - p1.y
                let length = hypot(dx, dy)

                guard length > 0.02 else { continue }

                let angle = atan2(dy, dx)
                let midpoint = CGPoint(x: (p1.x + p2.x) * 0.5, y: (p1.y + p2.y) * 0.5)

                candidates.append(
                    GoalLineCandidate(
                        start: p1,
                        end: p2,
                        length: length,
                        angle: angle,
                        midpoint: midpoint,
                        score: length
                    )
                )
            }
        }

        return candidates
    }

    private func selectBestTopBar(from candidates: [GoalLineCandidate]) -> GoalLineCandidate? {
        let horizontalTolerance: CGFloat = .pi / 7   // about 25 degrees

        return candidates
            .filter { candidate in
                let isHorizontal = abs(candidate.angle) < horizontalTolerance ||
                    abs(abs(candidate.angle) - .pi) < horizontalTolerance
                let isUpperHalf = candidate.midpoint.y < 0.7
                return isHorizontal && isUpperHalf
            }
            .max(by: { $0.score < $1.score })
    }

    private func selectBestPost(from candidates: [GoalLineCandidate],
                                side: GoalSide,
                                topBar: GoalLineCandidate) -> GoalLineCandidate? {
        let verticalTolerance: CGFloat = .pi / 7   // about 25 degrees

        return candidates
            .filter { candidate in
                let isVertical = abs(abs(candidate.angle) - (.pi / 2)) < verticalTolerance
                guard isVertical else { return false }

                switch side {
                case .left:
                    return candidate.midpoint.x < topBar.midpoint.x
                case .right:
                    return candidate.midpoint.x > topBar.midpoint.x
                }
            }
            .max(by: { $0.score < $1.score })
    }

    private func lowerEndpoint(of line: GoalLineCandidate) -> CGPoint {
        line.start.y > line.end.y ? line.start : line.end
    }

    private func intersection(_ l1: GoalLineCandidate, _ l2: GoalLineCandidate) -> CGPoint? {
        let x1 = l1.start.x, y1 = l1.start.y
        let x2 = l1.end.x, y2 = l1.end.y
        let x3 = l2.start.x, y3 = l2.start.y
        let x4 = l2.end.x, y4 = l2.end.y

        let denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        guard abs(denom) > 0.000001 else { return nil }

        let px = ((x1 * y2 - y1 * x2) * (x3 - x4) -
                  (x1 - x2) * (x3 * y4 - y3 * x4)) / denom

        let py = ((x1 * y2 - y1 * x2) * (y3 - y4) -
                  (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        return CGPoint(x: px, y: py)
    }

    private func updateDetectorDebugLabel(rawResults: [VNDetectedObjectObservation], chosenConfidence: VNConfidence) {
        let stageName: String
        switch setupStage {
        case .detectingGoal: stageName = "locating"
        case .detectingGoalPlacement: stageName = "placing"
        case .detectingSceneStability: stageName = "stabilizing"
        case .detectingGoalContours: stageName = "contouring"
        case .setupComplete: stageName = "complete"
        }
        let sorted = rawResults.sorted(by: { $0.confidence > $1.confidence })
        let topConf = sorted.first?.confidence ?? 0
        let secondConf = sorted.dropFirst().first?.confidence ?? 0
        let chosenText = chosenConfidence > 0 ? String(format: "%.2f", chosenConfidence) : "–"
        let text = String(format: " stage: %@\n raw: %d  top: %.2f  #2: %.2f\n chosen: %@  thresh: %.2f ",
                          stageName, rawResults.count, topConf, secondConf, chosenText, goalDetectionMinConfidence)
        DispatchQueue.main.async {
            self.detectorDebugLabel.text = text
        }
    }

    private func clampedToUnitRect(_ rect: CGRect) -> CGRect {
        let x = max(0, min(rect.origin.x, 1))
        let y = max(0, min(rect.origin.y, 1))
        let maxX = max(0, min(rect.maxX, 1))
        let maxY = max(0, min(rect.maxY, 1))
        return CGRect(x: x,
                      y: y,
                      width: max(0, maxX - x),
                      height: max(0, maxY - y))
    }
}

extension SetupViewController: GameStateChangeObserver {
    func gameManagerDidEnter(state: GameManager.State, from previousState: GameManager.State?) {
        switch state {
        case is GameManager.DetectedGoalState:
            setupStage =  .setupComplete
            statusLabel.text = "Goal Detected"
            statusLabel.perform(transitions: [.popUp, .popOut], durations: [0.25, 0.12], delayBetween: 1.5) {
                self.gameManager.stateMachine.enter(GameManager.DetectingPlayerState.self)
            }
        default:
            break
        }
    }
}
