/*
See LICENSE folder for this sample’s licensing information.

Abstract:
This is a collection of common data types, constants and helper functions used in the app.
*/

import UIKit
import Vision

enum KickType: String, CaseIterable {
    case instepKick = "Good! +" // Finesse
    case outsideKick = "Amazing!! +" // Trivela
    case insideKick = "Unique! +" // Power
    case none = "Nice! +" // Undefined
}

enum Scoring: Int {
    case zero = 0
    case one = 1
    case three = 3
    case five = 5
    case fifteen = 15
}
struct KickMetrics {
    var score = Scoring.zero
    var releaseSpeed = 0.0
    var releaseAngle = 0.0
    var kickType = KickType.none
    var finalShotLocation: CGPoint = .zero

    mutating func updateKickType(_ type: KickType) {
        kickType = type
    }

    mutating func updateFinalShotLocation(_ location: CGPoint) {
        finalShotLocation = location
    }

    mutating func updateMetrics(newScore: Scoring, speed: Double, angle: Double) {
        score = newScore
        releaseSpeed = speed
        releaseAngle = angle
    }
}

struct PlayerStats {
    var totalScore = 0
    var kickCount = 0
    var topSpeed = 0.0
    var avgSpeed = 0.0
    var releaseAngle = 0.0
    var avgReleaseAngle = 0.0
    var poseObservations = [VNHumanBodyPoseObservation]()
    var kickPaths = [CGPath]()
    
    mutating func reset() {
        topSpeed = 0
        avgSpeed = 0
        totalScore = 0
        kickCount = 0
        releaseAngle = 0
        poseObservations = []
    }

    mutating func resetObservations() {
        poseObservations = []
    }

    mutating func adjustMetrics(score: Scoring, speed: Double, releaseAngle: Double, kickType: KickType) {
        kickCount += 1
        totalScore += score.rawValue
        avgSpeed = (avgSpeed * Double(kickCount - 1) + speed) / Double(kickCount)
        avgReleaseAngle = (avgReleaseAngle * Double(kickCount - 1) + releaseAngle) / Double(kickCount)
        if speed > topSpeed {
            topSpeed = speed
        }
    }

    mutating func storePath(_ path: CGPath) {
        kickPaths.append(path)
    }

    mutating func storeObservation(_ observation: VNHumanBodyPoseObservation) {
        if poseObservations.count >= GameConstants.maxPoseObservations {
            poseObservations.removeFirst()
        }
        poseObservations.append(observation)
    }

    mutating func getReleaseAngle() -> Double {
        if !poseObservations.isEmpty {
            let observationCount = poseObservations.count
            let postReleaseObservationCount = GameConstants.trajectoryLength + GameConstants.maxTrajectoryInFlightPoseObservations
            let keyFrameForReleaseAngle = observationCount > postReleaseObservationCount ? observationCount - postReleaseObservationCount : 0
            let observation = poseObservations[keyFrameForReleaseAngle]
            let (rightHip, rightAnkle) = legJoints(for: observation)
            // Release angle is computed by measuring the angle leg (hip to ankle) makes with the horizontal
            releaseAngle = rightHip.angleFromHorizontal(to: rightAnkle)
        }
        return releaseAngle
    }

    mutating func getLastKickType() -> KickType {
        guard let actionClassifier = try? ShotClassifier(configuration: MLModelConfiguration()),
              let poseMultiArray = prepareInputWithObservations(poseObservations),
              let predictions = try? actionClassifier.prediction(poses: poseMultiArray),
              let kickType = KickType(rawValue: predictions.label.capitalized) else {
            return .none
        }
        return kickType
    }
}

struct GameConstants {
    static let maxKicks = 8
    static let newGameTimer = 5
    static let goalLength = 1.22
    static let trajectoryLength = 15
    static let maxPoseObservations = 45
    static let noObservationFrameLimit = 20
    static let maxDistanceWithCurrentTrajectory: CGFloat = 250
    static let maxTrajectoryInFlightPoseObservations = 10
}

let jointsOfInterest: [VNHumanBodyPoseObservation.JointName] = [
    .rightAnkle,
    .rightKnee,
    .rightHip,
    .leftAnkle,
    .leftKnee,
    .leftHip
]

func legJoints(for observation: VNHumanBodyPoseObservation) -> (CGPoint, CGPoint) {
    var rightHip = CGPoint(x: 0, y: 0)
    var rightAnkle = CGPoint(x: 0, y: 0)

    guard let identifiedPoints = try? observation.recognizedPoints(.all) else {
        return (rightHip, rightAnkle)
    }
    for (key, point) in identifiedPoints where point.confidence > 0.1 {
        switch key {
        case .rightHip:
            rightHip = point.location
        case .rightAnkle:
            rightAnkle = point.location
        default:
            break
        }
    }
    return (rightHip, rightAnkle)
}

func getBodyJointsFor(observation: VNHumanBodyPoseObservation) -> ([VNHumanBodyPoseObservation.JointName: CGPoint]) {
    var joints = [VNHumanBodyPoseObservation.JointName: CGPoint]()
    guard let identifiedPoints = try? observation.recognizedPoints(.all) else {
        return joints
    }
    for (key, point) in identifiedPoints {
        guard point.confidence > 0.1 else { continue }
        if jointsOfInterest.contains(key) {
            joints[key] = point.location
        }
    }
    return joints
}

// MARK: - Pipeline warmup

func warmUpVisionPipeline() {
    // In order to preload the models and all associated resources
    // we perform all Vision requests used in the app on a small image (we use one of the assets bundled with our app).
    // This allows to avoid any model loading/compilation costs later when we run these requests on real time video input.
    guard let image = #imageLiteral(resourceName: "Score1").cgImage,
          let detectorModel = try? GoalDetector(configuration: MLModelConfiguration()).model,
          let goalDetectionRequest = try? VNCoreMLRequest(model: VNCoreMLModel(for: detectorModel)) else {
        return
    }
    let bodyPoseRequest = VNDetectHumanBodyPoseRequest()
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    try? handler.perform([bodyPoseRequest, goalDetectionRequest])
}

// MARK: - Activity Classification Helpers

func prepareInputWithObservations(_ observations: [VNHumanBodyPoseObservation]) -> MLMultiArray? {
    let numAvailableFrames = observations.count
    let observationsNeeded = 120 // Changed from 45
    var multiArrayBuffer = [MLMultiArray]()

    for frameIndex in 0 ..< min(numAvailableFrames, observationsNeeded) {
        let pose = observations[frameIndex]
        do {
            let oneFrameMultiArray = try pose.keypointsMultiArray()
            multiArrayBuffer.append(oneFrameMultiArray)
        } catch {
            continue
        }
    }
    
    // If poseWindow does not have enough frames (120) yet, we need to pad 0s
    if numAvailableFrames < observationsNeeded {
        for _ in 0 ..< (observationsNeeded - numAvailableFrames) {
            do {
                let oneFrameMultiArray = try MLMultiArray(shape: [1, 3, 18], dataType: .double)
                try resetMultiArray(oneFrameMultiArray)
                multiArrayBuffer.append(oneFrameMultiArray)
            } catch {
                continue
            }
        }
    }
    return MLMultiArray(concatenating: [MLMultiArray](multiArrayBuffer), axis: 0, dataType: .float)
}

func resetMultiArray(_ predictionWindow: MLMultiArray, with value: Double = 0.0) throws {
    let pointer = try UnsafeMutableBufferPointer<Double>(predictionWindow)
    pointer.initialize(repeating: value)
}

// MARK: - Helper extensions

extension CGPoint {
    func distance(to point: CGPoint) -> CGFloat {
        return hypot(x - point.x, y - point.y)
    }
    
    func angleFromHorizontal(to point: CGPoint) -> Double {
        let angle = atan2(point.y - y, point.x - x)
        let deg = abs(angle * (180.0 / CGFloat.pi))
        return Double(round(100 * deg) / 100)
    }
}

extension CGAffineTransform {
    static var verticalFlip = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -1)
}

extension UIBezierPath {
    convenience init(cornersOfRect borderRect: CGRect, cornerSize: CGSize, cornerRadius: CGFloat) {
        self.init()
        let cornerSizeH = cornerSize.width
        let cornerSizeV = cornerSize.height
        // top-left
        move(to: CGPoint(x: borderRect.minX, y: borderRect.minY + cornerSizeV + cornerRadius))
        addLine(to: CGPoint(x: borderRect.minX, y: borderRect.minY + cornerRadius))
        addArc(withCenter: CGPoint(x: borderRect.minX + cornerRadius, y: borderRect.minY + cornerRadius),
               radius: cornerRadius,
               startAngle: CGFloat.pi,
               endAngle: -CGFloat.pi / 2,
               clockwise: true)
        addLine(to: CGPoint(x: borderRect.minX + cornerSizeH + cornerRadius, y: borderRect.minY))
        // top-right
        move(to: CGPoint(x: borderRect.maxX - cornerSizeH - cornerRadius, y: borderRect.minY))
        addLine(to: CGPoint(x: borderRect.maxX - cornerRadius, y: borderRect.minY))
        addArc(withCenter: CGPoint(x: borderRect.maxX - cornerRadius, y: borderRect.minY + cornerRadius),
               radius: cornerRadius,
               startAngle: -CGFloat.pi / 2,
               endAngle: 0,
               clockwise: true)
        addLine(to: CGPoint(x: borderRect.maxX, y: borderRect.minY + cornerSizeV + cornerRadius))
        // bottom-right
        move(to: CGPoint(x: borderRect.maxX, y: borderRect.maxY - cornerSizeV - cornerRadius))
        addLine(to: CGPoint(x: borderRect.maxX, y: borderRect.maxY - cornerRadius))
        addArc(withCenter: CGPoint(x: borderRect.maxX - cornerRadius, y: borderRect.maxY - cornerRadius),
               radius: cornerRadius,
               startAngle: 0,
               endAngle: CGFloat.pi / 2,
               clockwise: true)
        addLine(to: CGPoint(x: borderRect.maxX - cornerSizeH - cornerRadius, y: borderRect.maxY))
        // bottom-left
        move(to: CGPoint(x: borderRect.minX + cornerSizeH + cornerRadius, y: borderRect.maxY))
        addLine(to: CGPoint(x: borderRect.minX + cornerRadius, y: borderRect.maxY))
        addArc(withCenter: CGPoint(x: borderRect.minX + cornerRadius,
                                   y: borderRect.maxY - cornerRadius),
               radius: cornerRadius,
               startAngle: CGFloat.pi / 2,
               endAngle: CGFloat.pi,
               clockwise: true)
        addLine(to: CGPoint(x: borderRect.minX, y: borderRect.maxY - cornerSizeV - cornerRadius))
    }
}

// MARK: - Errors

enum AppError: Error {
    case captureSessionSetup(reason: String)
    case createRequestError(reason: String)
    case videoReadingError(reason: String)
    
    static func display(_ error: Error, inViewController viewController: UIViewController) {
        if let appError = error as? AppError {
            appError.displayInViewController(viewController)
        } else {
            print(error)
        }
    }
    
    func displayInViewController(_ viewController: UIViewController) {
        let title: String?
        let message: String?
        switch self {
        case .captureSessionSetup(let reason):
            title = "AVSession Setup Error"
            message = reason
        case .createRequestError(let reason):
            title = "Error Creating Vision Request"
            message = reason
        case .videoReadingError(let reason):
            title = "Error Reading Recorded Video."
            message = reason
        }
        
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        
        viewController.present(alert, animated: true)
    }
}
