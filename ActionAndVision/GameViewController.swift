/*
See LICENSE folder for this sample’s licensing information.

Abstract:
View controller responsible for the game flow.
     The game flow consists of the following tasks:
     - player detection
     - trajectory detection
     - player action classification
     - release angle, release speed and score computation
*/

import UIKit
import AVFoundation
import Vision

class GameViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    @IBOutlet weak var scoreLabel: UILabel!
    @IBOutlet weak var lastKickMetricsLabel: UILabel!
    @IBOutlet var shots: [UIImageView]!
    @IBOutlet weak var gameStatusLabel: OverlayLabel!
    @IBOutlet weak var kickTypeLabel: UILabel!
    @IBOutlet weak var releaseAngleLabel: UILabel!
    @IBOutlet weak var metricsStackView: UIStackView!
    @IBOutlet weak var speedLabel: UILabel!
    @IBOutlet weak var speedStackView: UIStackView!
    @IBOutlet weak var kickTypeImage: UIImageView!
    @IBOutlet weak var dashboardView: DashboardView!
    @IBOutlet weak var goodKickView: ProgressView!
    @IBOutlet weak var greatKickView: ProgressView!
    @IBOutlet weak var amazingKickView: ProgressView!
    private let gameManager = GameManager.shared
    private let detectPlayerRequest = VNDetectHumanBodyPoseRequest()
    private var playerDetected = false
    private var isShotInTargetRegion = false
    private var kickRegion = CGRect.null
    private var targetRegion = CGRect.null
    private let trajectoryView = TrajectoryView()
    private let playerBoundingBox = BoundingBoxView()
    private let jointSegmentView = JointSegmentView()
    private var noObservationFrameCount = 0
    private var trajectoryInFlightPoseObservations = 0
    private var showSummaryGesture: UITapGestureRecognizer!
    private let trajectoryQueue = DispatchQueue(label: "com.ActionAndVision.trajectory", qos: .userInteractive)
    private let bodyPoseDetectionMinConfidence: VNConfidence = 0.6
    private let trajectoryDetectionMinConfidence: VNConfidence = 0.9
    private let bodyPoseRecognizedPointMinConfidence: VNConfidence = 0.1
    private lazy var detectTrajectoryRequest: VNDetectTrajectoriesRequest! =
    VNDetectTrajectoriesRequest(frameAnalysisSpacing: .zero, trajectoryLength: GameConstants.trajectoryLength)
    
    //Variables - KPIs
    var lastKickMetrics: KickMetrics {
        get {
            return gameManager.lastKickMetrics
        }
        set {
            gameManager.lastKickMetrics = newValue
        }
    }
    
    var playerStats: PlayerStats {
        get {
            return gameManager.playerStats
        }
        set {
            gameManager.playerStats = newValue
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setUIElements()
        showSummaryGesture = UITapGestureRecognizer(target: self, action: #selector(handleShowSummaryGesture(_:)))
        showSummaryGesture.numberOfTapsRequired = 2
        view.addGestureRecognizer(showSummaryGesture)
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        gameStatusLabel.perform(transition: .fadeIn, duration: 0.25)
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        detectTrajectoryRequest = nil
    }
    
    func getScoreLabelAttributedStringForScore(_ score: Int) -> NSAttributedString {
        let totalScore = NSMutableAttributedString(string: " Score: ", attributes: [.foregroundColor: #colorLiteral(red: 0.501960814, green: 0.501960814, blue: 0.501960814, alpha: 1)])
        totalScore.append(NSAttributedString(string: "\(score)", attributes: [.foregroundColor: #colorLiteral(red: 0.501960814, green: 0.501960814, blue: 0.501960814, alpha: 1)]))
        totalScore.append(NSAttributedString(string: "", attributes: [.foregroundColor: #colorLiteral(red: 0.501960814, green: 0.501960814, blue: 0.501960814, alpha: 1)]))
        return totalScore
    }
    
    func setUIElements() {
        resetKPILabels()
        playerBoundingBox.borderColor = #colorLiteral(red: 1, green: 1, blue: 1, alpha: 1)
        playerBoundingBox.backgroundOpacity = 0
        playerBoundingBox.isHidden = true
        view.addSubview(playerBoundingBox)
        view.addSubview(jointSegmentView)
        view.addSubview(trajectoryView)
        gameStatusLabel.text = "Waiting for player"
        // Set kick type counters
        goodKickView.throwType = .insideKick
        greatKickView.throwType = .instepKick
        amazingKickView.throwType = .outsideKick
        scoreLabel.attributedText = getScoreLabelAttributedStringForScore(0)
    }
    
    func resetKPILabels() {
        // Reset Speed and kickType image
        dashboardView.speed = 0
        kickTypeImage.image = nil
        // Hode KPI labels
        dashboardView.isHidden = true
        speedStackView.isHidden = true
        metricsStackView.isHidden = true
    }
    
    func updateKPILabels() {
        // Show KPI labels
        dashboardView.isHidden = false
        speedStackView.isHidden = false
        metricsStackView.isHidden = false
        // Update text for KPI labels
        speedLabel.text = "\(lastKickMetrics.releaseSpeed)"
        kickTypeLabel.text = lastKickMetrics.kickType.rawValue.capitalized
        releaseAngleLabel.text = "\(lastKickMetrics.releaseAngle)°"
        lastKickMetricsLabel.text = "\(lastKickMetrics.score.rawValue)"
        // Update score label
        scoreLabel.attributedText = getScoreLabelAttributedStringForScore(gameManager.playerStats.totalScore)
        // Update kick type image
        kickTypeImage.image = UIImage(named: lastKickMetrics.kickType.rawValue)
        // Update kick type counters
        switch lastKickMetrics.kickType {
        case .instepKick:
            greatKickView.incrementKickCount()
        case .insideKick:
            goodKickView.incrementKickCount()
        case .outsideKick:
            amazingKickView.incrementKickCount()
        default:
            break
        }
        // Update score for shot views
        let shotView = shots[playerStats.kickCount - 1]
        shotView.image = UIImage(named: "Score\(lastKickMetrics.score.rawValue)")
        // Hide KPI labels after 2 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            self.clearKPILabels()
        }
    }

    func clearKPILabels() {
        // Hide KPI labels
        dashboardView.isHidden = true
        speedStackView.isHidden = true
        metricsStackView.isHidden = true
        // Optionally clear text for KPI labels (if needed)
        //speedLabel.text = ""
        //kickTypeLabel.text = ""
        //releaseAngleLabel.text = ""
        //scoreLabel.text = ""
        //kickTypeImage.image = nil
    }
    
    func updateBoundingBox(_ boundingBox: BoundingBoxView, withRect rect: CGRect?) {
        // Update the frame for player bounding box
        boundingBox.frame = rect ?? .zero
        boundingBox.perform(transition: (rect == nil ? .fadeOut : .fadeIn), duration: 0.1)
    }
    
    func humanBoundingBox(for observation: VNHumanBodyPoseObservation) -> CGRect {
        var box = CGRect.zero
        var normalizedBoundingBox = CGRect.null
        // Process body points only if the confidence is high.
        guard observation.confidence > bodyPoseDetectionMinConfidence, let points = try? observation.recognizedPoints(forGroupKey: .all) else {
            return box
        }
        // Only use point if human pose joint was detected reliably.
        for (_, point) in points where point.confidence > bodyPoseRecognizedPointMinConfidence {
            normalizedBoundingBox = normalizedBoundingBox.union(CGRect(origin: point.location, size: .zero))
        }
        if !normalizedBoundingBox.isNull {
            box = normalizedBoundingBox
        }
        // Fetch body joints from the observation and overlay them on the player.
        let joints = getBodyJointsFor(observation: observation)
        DispatchQueue.main.async {
            self.jointSegmentView.joints = joints
            self.playerBoundingBox.alpha = 0.0
            self.jointSegmentView.alpha = 0
        }
        // Store the body pose observation in playerStats when the game is in TrackKicksState.
        // We will use these observations for action classification once the throw is complete.
        if gameManager.stateMachine.currentState is GameManager.TrackKicksState {
            playerStats.storeObservation(observation)
            if trajectoryView.inFlight {
                trajectoryInFlightPoseObservations += 1
            }
        }
        return box
    }
    
    // Define regions to filter relavant trajectories for the game
    // kickRegion: Region to the right of the player to detect start of kick
    // targetRegion: Region around the goal to determine end of kick
    func resetTrajectoryRegions() {
        let goalRegion = gameManager.goalRegion
        let playerRegion = playerBoundingBox.frame
        let kickWindowXBuffer: CGFloat = 50
        let kickWindowYBuffer: CGFloat = 50
        let targetWindowXBuffer: CGFloat = 50
        let kickRegionWidth: CGFloat = 400
        kickRegion = CGRect(x: playerRegion.maxX + kickWindowXBuffer, y: 0, width: kickRegionWidth, height: playerRegion.maxY - kickWindowYBuffer)
        targetRegion = CGRect(x: goalRegion.minX - targetWindowXBuffer, y: 0,
                              width: goalRegion.width + 2 * targetWindowXBuffer, height: goalRegion.maxY)
    }
    
    // Adjust the kickRegion based on location of the bag.
    // Move the kickRegion to the right until we reach the target region.
    func updateTrajectoryRegions() {
        let trajectoryLocation = trajectoryView.fullTrajectory.currentPoint
        let didShotCrossCenterOfKickRegion = trajectoryLocation.x > kickRegion.origin.x + kickRegion.width / 2
        guard !(kickRegion.contains(trajectoryLocation) && didShotCrossCenterOfKickRegion) else {
            return
        }
        // Overlap buffer window between kickRegion and targetRegion
        let overlapWindowBuffer: CGFloat = 50
        if targetRegion.contains(trajectoryLocation) {
            // When shot is in target region, set the kickRegion to targetRegion.
            kickRegion = targetRegion
        } else if trajectoryLocation.x + kickRegion.width / 2 - overlapWindowBuffer < targetRegion.origin.x {
            // Move the kickRegion forward to have the shot at the center.
            kickRegion.origin.x = trajectoryLocation.x - kickRegion.width / 2
        }
        trajectoryView.roi = kickRegion
    }
    
    func processTrajectoryObservations(_ controller: CameraViewController, _ results: [VNTrajectoryObservation]) {
        if self.trajectoryView.inFlight && results.count < 1 {
            // The trajectory is already in flight but VNDetectTrajectoriesRequest doesn't return any trajectory observations.
            self.noObservationFrameCount += 1
            if self.noObservationFrameCount > GameConstants.noObservationFrameLimit {
                // Ending the kick as we don't see any observations in consecutive GameConstants.noObservationFrameLimit frames.
                self.updatePlayerStats(controller)
            }
        } else {
            for path in results where path.confidence > trajectoryDetectionMinConfidence {
                // VNDetectTrajectoriesRequest has returned some trajectory observations.
                // Process the path only when the confidence is over 90%.
                self.trajectoryView.duration = path.timeRange.duration.seconds
                self.trajectoryView.points = path.detectedPoints
                self.trajectoryView.perform(transition: .fadeIn, duration: 0.25)
                if !self.trajectoryView.fullTrajectory.isEmpty {
                    // Hide the previous kick metrics once a new kick is detected.
                    if !self.dashboardView.isHidden {
                        self.resetKPILabels()
                    }
                    self.updateTrajectoryRegions()
                    if self.trajectoryView.isThrowComplete {
                        // Update the player statistics once the kick is complete.
                        self.updatePlayerStats(controller)
                    }
                }
                self.noObservationFrameCount = 0
            }
        }
    }
    
    func updatePlayerStats(_ controller: CameraViewController) {
        let finalShotLocation = trajectoryView.finalShotLocation
        playerStats.storePath(self.trajectoryView.fullTrajectory.cgPath)
        trajectoryView.resetPath()
        lastKickMetrics.updateKickType(playerStats.getLastKickType())
        let score = computeScore(controller.viewPointForVisionPoint(finalShotLocation))
        // Compute the speed in mph
        // trajectoryView.speed is in points/second, convert that to meters/second by multiplying the pointToMeterMultiplier.
        // 1 meters/second = 2.24 miles/hour
        let releaseSpeed = round(trajectoryView.speed * gameManager.pointToMeterMultiplier * 2.24) / 100
        let releaseAngle = playerStats.getReleaseAngle()
        lastKickMetrics.updateMetrics(newScore: score, speed: releaseSpeed, angle: releaseAngle)
        self.gameManager.stateMachine.enter(GameManager.KickCompletedState.self)
    }
    
    func computeScore(_ finalShotLocation: CGPoint) -> Scoring {
        let heightBuffer: CGFloat = 100
        let goalRegion = gameManager.goalRegion
        // In some cases trajectory observation may not end exactly in the goal and end a few pixels above the goal.
        // This can happen especially when the shot bounces into the goal. Filtering conditions can be adjusted to get those observations as well.
        // Defining extended regions for goal and the top corner with a heightBuffer to cover these cases.
        let extendedGoalRegion = CGRect(x: goalRegion.origin.x, y: goalRegion.origin.y - heightBuffer,
                                         width: goalRegion.width, height: goalRegion.height + heightBuffer)
        let TopCornerRegion = gameManager.TopCornerRegion
        let extendedTopCornerRegion = CGRect(x: TopCornerRegion.origin.x, y: TopCornerRegion.origin.y - heightBuffer,
                                        width: TopCornerRegion.width, height: TopCornerRegion.height + heightBuffer)
        if !extendedGoalRegion.contains(finalShotLocation) {
            // Shot missed the goal
            return Scoring.zero
        } else if extendedTopCornerRegion.contains(finalShotLocation) {
            // Shot landed in the top corner
            return lastKickMetrics.kickType == .outsideKick ? Scoring.fifteen : Scoring.three
        } else {
            // Shot landed in the goal
            return lastKickMetrics.kickType == .outsideKick ? Scoring.five : Scoring.one
        }
    }
}

extension GameViewController: GameStateChangeObserver {
    func gameManagerDidEnter(state: GameManager.State, from previousState: GameManager.State?) {
        switch state {
        case is GameManager.DetectedPlayerState:
            playerDetected = true
            playerStats.reset()
            playerBoundingBox.perform(transition: .fadeOut, duration: 1.0)
            gameStatusLabel.text = "Go"
            gameStatusLabel.perform(transitions: [.popUp, .popOut], durations: [0.25, 0.12], delayBetween: 1) {
                self.gameManager.stateMachine.enter(GameManager.TrackKicksState.self)
            }
        case is GameManager.TrackKicksState:
            resetTrajectoryRegions()
            trajectoryView.roi = kickRegion
        case is GameManager.KickCompletedState:
            dashboardView.speed = lastKickMetrics.releaseSpeed
            dashboardView.animateSpeedChart()
            playerStats.adjustMetrics(score: lastKickMetrics.score, speed: lastKickMetrics.releaseSpeed,
                                      releaseAngle: lastKickMetrics.releaseAngle, kickType: lastKickMetrics.kickType)
            playerStats.resetObservations()
            trajectoryInFlightPoseObservations = 0
            self.updateKPILabels()
            
            // Display the "Finesse" image in the gameStatusLabel
            gameStatusLabel.attributedText = {
                let finesseImage = UIImage(named: "Finesse")
                let imageAttachment = NSTextAttachment()
                imageAttachment.image = finesseImage
                
                return NSAttributedString(attachment: imageAttachment)
            }()
            
            gameStatusLabel.perform(transitions: [.popUp, .popOut], durations: [0.25, 0.12], delayBetween: 1) {
                if self.playerStats.kickCount == GameConstants.maxKicks {
                    self.gameManager.stateMachine.enter(GameManager.ShowSummaryState.self)
                } else {
                    self.gameManager.stateMachine.enter(GameManager.TrackKicksState.self)
                }
            }
        default:
            break
        }
    }
}

extension GameViewController: CameraViewControllerOutputDelegate {
    func cameraViewController(_ controller: CameraViewController, didReceiveBuffer buffer: CMSampleBuffer, orientation: CGImagePropertyOrientation) {
        let visionHandler = VNImageRequestHandler(cmSampleBuffer: buffer, orientation: orientation, options: [:])
        if gameManager.stateMachine.currentState is GameManager.TrackKicksState {
            DispatchQueue.main.async {
                // Get the frame of rendered view
                let normalizedFrame = CGRect(x: 0, y: 0, width: 1, height: 1)
                self.jointSegmentView.frame = controller.viewRectForVisionRect(normalizedFrame)
                self.trajectoryView.frame = controller.viewRectForVisionRect(normalizedFrame)
            }
            // Perform the trajectory request in a separate dispatch queue.
            trajectoryQueue.async {
                do {
                    try visionHandler.perform([self.detectTrajectoryRequest])
                    if let results = self.detectTrajectoryRequest.results {
                        DispatchQueue.main.async {
                            self.processTrajectoryObservations(controller, results)
                        }
                    }
                } catch {
                    AppError.display(error, inViewController: self)
                }
            }
        }
        // Body pose request is performed on the same camera queue to ensure the highlighted joints are aligned with the player.
        // Run bodypose request for additional GameConstants.maxPostReleasePoseObservations frames after the first trajectory observation is detected.
        if !(self.trajectoryView.inFlight && self.trajectoryInFlightPoseObservations >= GameConstants.maxTrajectoryInFlightPoseObservations) {
            do {
                try visionHandler.perform([detectPlayerRequest])
                if let result = detectPlayerRequest.results?.first {
                    let box = humanBoundingBox(for: result)
                    let boxView = playerBoundingBox
                    DispatchQueue.main.async {
                        let inset: CGFloat = -20.0
                        let viewRect = controller.viewRectForVisionRect(box).insetBy(dx: inset, dy: inset)
                        self.updateBoundingBox(boxView, withRect: viewRect)
                        if !self.playerDetected && !boxView.isHidden {
                            self.gameStatusLabel.alpha = 0
                            self.resetTrajectoryRegions()
                            self.gameManager.stateMachine.enter(GameManager.DetectedPlayerState.self)
                        }
                    }
                }
            } catch {
                AppError.display(error, inViewController: self)
            }
        } else {
            // Hide player bounding box
            DispatchQueue.main.async {
                if !self.playerBoundingBox.isHidden {
                    self.playerBoundingBox.isHidden = true
                    self.jointSegmentView.resetView()
                }
            }
        }
    }
}

extension GameViewController {
    @objc
    func handleShowSummaryGesture(_ gesture: UITapGestureRecognizer) {
        if gesture.state == .ended {
            self.gameManager.stateMachine.enter(GameManager.ShowSummaryState.self)
        }
    }
}
