/*
See LICENSE folder for this sample’s licensing information.

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
 
    private let gameManager = GameManager.shared
    private let GoalLocationGuide = BoundingBoxView()
    private let goalBoundingBox = BoundingBoxView()

    private var GoalDetectionRequest: VNCoreMLRequest!
    private let GoalDetectionMinConfidence: VNConfidence = 0.1 // was 0.6
    
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
        GoalLocationGuide.borderColor = #colorLiteral(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
        GoalLocationGuide.borderWidth = 3
        GoalLocationGuide.borderCornerRadius = 4
        GoalLocationGuide.borderCornerSize = 30
        GoalLocationGuide.backgroundOpacity = 0.25
        GoalLocationGuide.isHidden = true
        view.addSubview(GoalLocationGuide)
        goalBoundingBox.borderColor = #colorLiteral(red: 1, green: 0.5763723254, blue: 0, alpha: 1)
        goalBoundingBox.borderWidth = 2
        goalBoundingBox.borderCornerRadius = 4
        goalBoundingBox.borderCornerSize = 0
        goalBoundingBox.backgroundOpacity = 0.45
        goalBoundingBox.isHidden = true
        view.addSubview(goalBoundingBox)
        updateSetupState()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        do {
            // Create Vision request based on CoreML model
            let model = try VNCoreMLModel(for: GoalDetector(configuration: MLModelConfiguration()).model)
            GoalDetectionRequest = VNCoreMLRequest(model: model)
            // Since board is close to the side of a landscape image,
            // we need to set crop and scale option to scaleFit.
            // By default vision request will run on centerCrop.
            GoalDetectionRequest.imageCropAndScaleOption = .scaleFit
        } catch {
            let error = AppError.createRequestError(reason: "Could not create Vision request for goal detector")
            AppError.display(error, inViewController: self)
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
        let GoalBox = goalBoundingBox
        DispatchQueue.main.async {
            switch self.setupStage {
            case .detectingGoal:
                self.statusLabel.text = "Locating Goal"
                self.statusLabel.textColor = #colorLiteral(red: 0.501960814, green: 0.501960814, blue: 0.501960814, alpha: 1)
            case .detectingGoalPlacement:
                // Goal placement guide is shown only when using camera feed.
                // Otherwise we always assume the board is placed correctly.
                var boxPlacedCorrectly = true
                if !self.GoalLocationGuide.isHidden {
                    boxPlacedCorrectly = GoalBox.containedInside(self.GoalLocationGuide)
                }
                GoalBox.borderColor = boxPlacedCorrectly ? #colorLiteral(red: 0.4641711116, green: 1, blue: 0, alpha: 1) : #colorLiteral(red: 1, green: 0.5763723254, blue: 0, alpha: 1)
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

    func analyzeGoalContours(_ contours: [VNContour]) -> (crossbarPath: CGPath, topCornerPath: CGPath)? {
        // Simplify contours and ignore resulting contours with less than 3 points.
        let polyContours = contours.compactMap { (contour) -> VNContour? in
            guard let polyContour = try? contour.polygonApproximation(epsilon: 0.01),
                  polyContour.pointCount >= 3 else {
                return nil
            }
            return polyContour
        }
        // Goal contour is the contour with the largest amount of points.
        guard let GoalContour = polyContours.max(by: { $0.pointCount < $1.pointCount }) else {
            return nil
        }
        // First, find the crossbar, which is the longest diagonal segment of the contour
        // located in the top part of the goal's bounding box.
        let contourPoints = GoalContour.normalizedPoints.map { return CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
        let diagonalThreshold = CGFloat(0.02)
        var largestDiff = CGFloat(0.0)
        let goalPath = UIBezierPath()
        let countLessOne = contourPoints.count - 1
        // Both points should be in the top 2/3rds of the goal's bounding box.
        // Additionally one of them should be in the left half
        // and the other on in the right half of the goal's bounding box.
        for (point1, point2) in zip(contourPoints.prefix(countLessOne), contourPoints.suffix(countLessOne)) where
            min(point1.x, point2.x) < 0.5 && max(point1.x, point2.x) > 0.5 && point1.y >= 0.3 && point2.y >= 0.3 {
            let diffX = abs(point1.x - point2.x)
            let diffY = abs(point1.y - point2.y)
            guard diffX > diagonalThreshold && diffY > diagonalThreshold else {
                // This is not a diagonal line, skip this segment.
                continue
            }
            if diffX + diffY > largestDiff {
                largestDiff = diffX + diffY
                goalPath.removeAllPoints()
                goalPath.move(to: point1)
                goalPath.addLine(to: point2)
            }
        }
        guard largestDiff > 0 else {
            return nil
        }
        // Find the top corner contour, which is a polygon with 4 points and has a diagonal right edge.
        var topCornerPath: CGPath?
        for contour in polyContours where contour != GoalContour {
            let contourPoints = contour.normalizedPoints.map { return CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
            // Check if the contour has 4 points
            if contourPoints.count == 4 {
                // Check if the right edge is diagonal (comparing last two points)
                let point1 = contourPoints[2] // third point (left side of right edge)
                let point2 = contourPoints[3] // fourth point (right side of right edge)
                
                let diffX = abs(point1.x - point2.x)
                let diffY = abs(point1.y - point2.y)
                
                // Ensure the right edge is diagonal by checking threshold
                if diffX > diagonalThreshold && diffY > diagonalThreshold {
                    // If the right edge is diagonal, consider this the side netting
                    topCornerPath = contour.normalizedPath
                    break
                }
            }
        }
        // Return nil if we failed to find top corner.
        guard let detectedTopCornerPath = topCornerPath else {
            return nil
        }
        
        return (goalPath.cgPath, detectedTopCornerPath)
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
        try visionHandler.perform([GoalDetectionRequest])
        var rect: CGRect?
        var visionRect = CGRect.null
        if let results = GoalDetectionRequest.results as? [VNDetectedObjectObservation] {
            // Filter out classification results with low confidence
            let filteredResults = results.filter { $0.confidence > GoalDetectionMinConfidence }
            // Since the model is trained to detect only one object class (the goal)
            // there is no need to look at labels. If there is at least one result - we got the goal.
            if !filteredResults.isEmpty {
                visionRect = filteredResults[0].boundingBox
                rect = controller.viewRectForVisionRect(visionRect)
            }
        }
        // Show goal placement guide only when using camera feed.
        if gameManager.recordedVideoSource == nil {
            let guideVisionRect = CGRect(x: 0.7, y: 0.3, width: 0.28, height: 0.3)
            let guideRect = controller.viewRectForVisionRect(guideVisionRect)
            updateBoundingBox(GoalLocationGuide, withViewRect: guideRect, visionRect: guideVisionRect)
        }
        updateBoundingBox(goalBoundingBox, withViewRect: rect, visionRect: visionRect)
        // If rect is nil we need to keep looking for the board, otherwise check the goal placement
        self.setupStage = (rect == nil) ? .detectingGoal : .detectingGoalPlacement
    }
    
    private func detectGoalContours(_ controller: CameraViewController, _ buffer: CMSampleBuffer, _ orientation: CGImagePropertyOrientation) throws {
        let visionHandler = VNImageRequestHandler(cmSampleBuffer: buffer, orientation: orientation, options: [:])
        let contoursRequest = VNDetectContoursRequest()
        contoursRequest.contrastAdjustment = 1.6 // the default contrast is 2.0 but in this case 1.6 gives us more reliable results
        contoursRequest.regionOfInterest = goalBoundingBox.visionRect
        try visionHandler.perform([contoursRequest])
        if let result = contoursRequest.results?.first as? VNContoursObservation {
            // Perform analysis of the top level contours in order to find crossbar and top corner.
            guard let subpaths = analyzeGoalContours(result.topLevelContours) else {
                return
            }
            DispatchQueue.main.sync {
                // Save goal region
                self.gameManager.goalRegion = goalBoundingBox.frame
                // Calculate goal length based on the bounding box of the crossbar.
                let crossbarNormalizedBB = subpaths.crossbarPath.boundingBox
                // Convert normalized bounding box size to points.
                let crossbarSize = CGSize(width: crossbarNormalizedBB.width * goalBoundingBox.frame.width,
                                      height: crossbarNormalizedBB.height * goalBoundingBox.frame.height)
                // Calculate the length of the crossbar in points.
                let GoalLength = hypot(crossbarSize.width, crossbarSize.height)
                // Divide goal length in meters by goal length in points.
                self.gameManager.pointToMeterMultiplier = GameConstants.goalLength / Double(GoalLength)
                if let imageBuffer = CMSampleBufferGetImageBuffer(buffer) {
                    let imageData = CIImage(cvImageBuffer: imageBuffer).oriented(orientation)
                    self.gameManager.previewImage = UIImage(ciImage: imageData)
                }
                // Get the bounding box of top corner in CoreGraphics coordinates
                // and convert it to Vision coordinates by flipping vertically.
                var topCornerRect = subpaths.topCornerPath.boundingBox
                topCornerRect.origin.y = 1 - topCornerRect.origin.y - topCornerRect.height
                // Because we used region of interest in the request above,
                // the normalized coordinates of the returned contours are relative to that region.
                // Convert top corner region to Vision coordinates of entire image
                let goalRect = goalBoundingBox.visionRect
                let normalizedTopCornerRegion = CGRect(
                        x: goalRect.origin.x + topCornerRect.origin.x * goalRect.width,
                        y: goalRect.origin.y + topCornerRect.origin.y * goalRect.height,
                        width: topCornerRect.width * goalRect.width,
                        height: topCornerRect.height * goalRect.height)
                // Now convert top corner region from normalized Vision coordinates
                // to UIKit coordinates and save it.
                self.gameManager.TopCornerRegion = controller.viewRectForVisionRect(normalizedTopCornerRegion)
                // Combine crossbar and top corner paths to highlight them on the screen.
                let highlightPath = UIBezierPath(cgPath: subpaths.crossbarPath)
                highlightPath.append(UIBezierPath(cgPath: subpaths.topCornerPath))
                goalBoundingBox.visionPath = highlightPath.cgPath
                goalBoundingBox.borderColor = #colorLiteral(red: 1, green: 1, blue: 1, alpha: 0.199807363)
                self.gameManager.stateMachine.enter(GameManager.DetectedGoalState.self)
            }
        }
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
