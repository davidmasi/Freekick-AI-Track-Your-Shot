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
    private let goalLocationGuide = BoundingBoxView()
    private let goalBoundingBox = BoundingBoxView()

    private var goalDetectionRequest: VNCoreMLRequest!
    private let goalDetectionMinConfidence: VNConfidence = 0.1 // was 0.6
    
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
                self.statusLabel.text = "Locating Goal"
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

    func analyzeGoalContours(_ contours: [VNContour]) -> CGPath? {
        let path = UIBezierPath()

        // Compute bounding boxes manually
        let filteredContours = contours.compactMap { contour -> (VNContour, CGRect)? in
            let points = contour.normalizedPoints.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
            guard let minX = points.map({ $0.x }).min(),
                  let maxX = points.map({ $0.x }).max(),
                  let minY = points.map({ $0.y }).min(),
                  let maxY = points.map({ $0.y }).max() else {
                return nil
            }
            
            let boundingBox = CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
            return (contour, boundingBox)
        }

        // Find crossbar and posts
        var crossbar: VNContour?
        var leftPost: VNContour?
        var rightPost: VNContour?

        for (contour, boundingBox) in filteredContours {
            let aspectRatio = boundingBox.width / boundingBox.height

            if aspectRatio > 2.5 {
                // It's a long horizontal line -> Crossbar
                crossbar = contour
            } else if boundingBox.minX < 0.3 {
                // It's on the left side -> Left post
                leftPost = contour
            } else if boundingBox.maxX > 0.7 {
                // It's on the right side -> Right post
                rightPost = contour
            }
        }

        // Add detected parts to path
        if let crossbar = crossbar {
            path.append(UIBezierPath(cgPath: crossbar.normalizedPath))
        }
        if let leftPost = leftPost {
            path.append(UIBezierPath(cgPath: leftPost.normalizedPath))
        }
        if let rightPost = rightPost {
            path.append(UIBezierPath(cgPath: rightPost.normalizedPath))
        }

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
        if let results = goalDetectionRequest.results as? [VNDetectedObjectObservation] {
            // Filter out classification results with low confidence
            let filteredResults = results.filter { $0.confidence > goalDetectionMinConfidence }
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
            updateBoundingBox(goalLocationGuide, withViewRect: guideRect, visionRect: guideVisionRect)
        }
        updateBoundingBox(goalBoundingBox, withViewRect: rect, visionRect: visionRect)
        // If rect is nil we need to keep looking for the board, otherwise check the goal placement
        self.setupStage = (rect == nil) ? .detectingGoal : .detectingGoalPlacement
    }
    
    private func detectGoalContours(_ controller: CameraViewController, _ buffer: CMSampleBuffer, _ orientation: CGImagePropertyOrientation) throws {
        let visionHandler = VNImageRequestHandler(cmSampleBuffer: buffer, orientation: orientation, options: [:])
        let contoursRequest = VNDetectContoursRequest()
        contoursRequest.contrastAdjustment = 1.6 // Adjust contrast for better results
        contoursRequest.regionOfInterest = goalBoundingBox.visionRect
        
        try visionHandler.perform([contoursRequest])
        
        if let result = contoursRequest.results?.first as? VNContoursObservation {
            // Analyze detected contours
            guard let goalFramePath = analyzeGoalContours(result.topLevelContours) else {
                return
            }
            
            DispatchQueue.main.sync {
                // Save goal region
                self.gameManager.goalRegion = goalBoundingBox.frame
                
                // Calculate goal length based on the bounding box of the detected frame
                let crossbarNormalizedBB = goalFramePath.boundingBox
                let crossbarSize = CGSize(width: crossbarNormalizedBB.width * goalBoundingBox.frame.width,
                                          height: crossbarNormalizedBB.height * goalBoundingBox.frame.height)
                let goalLength = hypot(crossbarSize.width, crossbarSize.height)
                self.gameManager.pointToMeterMultiplier = GameConstants.goalLength / Double(goalLength)
                
                // Save the preview image for reference
                if let imageBuffer = CMSampleBufferGetImageBuffer(buffer) {
                    let imageData = CIImage(cvImageBuffer: imageBuffer).oriented(orientation)
                    self.gameManager.previewImage = UIImage(ciImage: imageData)
                }
                
                // Highlight the detected goal frame
                goalBoundingBox.visionPath = goalFramePath
                goalBoundingBox.borderColor = #colorLiteral(red: 1, green: 1, blue: 1, alpha: 0.199807363)
                
                // Move the game state forward
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
