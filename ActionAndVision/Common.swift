/*
See LICENSE folder for this sample’s licensing information.

Abstract:
This is a collection of common data types, constants and helper functions used in the app.
*/

import UIKit
import Vision

enum KickType: String, CaseIterable {
    case trivela  = "Amazing"   // outside kick
    case instep   = "Great"     // instep / finesse
    case laces    = "Good"      // power / laces
    case negative = "Negative"  // undefined

    var displayText: String {
        switch self {
        case .trivela:  return "Trivela "
        case .instep:   return "Instep  "
        case .laces:    return "Laces   "
        case .negative: return "Null    "
        }
    }

    var imageName: String {
        switch self {
        case .trivela:  return "Trivela"
        case .instep:   return "Finesse"
        case .laces:    return "Power"
        case .negative: return "Image"
        }
    }
}

enum Scoring: Int {
    case zero = 0
    case one = 1
    case three = 3
    case five = 5
    case fifteen = 15
}

enum ShotDirection {
    case left
    case right
}

struct KickMetrics {
    var score = Scoring.zero
    var releaseAngle = 0.0
    var kickType = KickType.negative
    var finalShotLocation: CGPoint = .zero
    var kickProbabilities: [String: Double] = [:]
    var observationCount = 0

    // Three velocities in MPH — populated in GameViewController.updatePlayerStats after depth
    // correction. Used by the criteria classifier (Phase 3) and the stats card UI (Phase 5).
    var exitVelocity = 0.0
    var totalVelocity = 0.0
    var finalVelocity = 0.0

    /// The number we show the user. Vision's 5-point exit sample sometimes lands on a slow patch
    /// (contact frames missed) even when the shot's true peak IS in the tracked data. Total-flight
    /// average catches those cases. max() picks whichever window better captured the shot; if
    /// they agree, it just equals exit. Never dragged down by the noisy final-burst reading.
    var topSpeed: Double { max(exitVelocity, totalVelocity) }

    // Total flight time in seconds (sum of trajectory burst durations from Vision).
    var flightTimeSeconds = 0.0
    // Estimated player→goal distance in meters at the moment the shot was taken. nil when the
    // calc couldn't be trusted (missing goal detection, no reliable player height, or the result
    // was outside a sanity band). Formatting via formatDistance() falls back to the standard
    // 18-yard default when nil.
    var distanceToGoalMeters: Double?

    mutating func updateKickType(_ type: KickType) {
        kickType = type
    }

    mutating func updateClassifierOutput(type: KickType, probabilities: [String: Double], observationCount: Int) {
        self.kickType = type
        self.kickProbabilities = probabilities
        self.observationCount = observationCount
    }

    mutating func updateFinalShotLocation(_ location: CGPoint) {
        finalShotLocation = location
    }

    // Speed is intentionally omitted — exitVelocity is now the sole speed of record and gets
    // set via updateVelocities(). Score and angle are the only two things updateMetrics touches.
    mutating func updateScoreAndAngle(newScore: Scoring, angle: Double) {
        score = newScore
        releaseAngle = angle
    }

    mutating func updateVelocities(exit: Double, total: Double, final: Double) {
        exitVelocity = exit
        totalVelocity = total
        finalVelocity = final
    }

    mutating func updateFlightTime(_ seconds: Double) {
        flightTimeSeconds = seconds
    }

    mutating func updateDistance(_ meters: Double?) {
        distanceToGoalMeters = meters
    }

    // Pose-derived criteria for classification. nil = couldn't determine (missing pose data or
    // low joint confidence). Phase 3 treats nil as "unavailable" and shifts to fallback mode.
    var bodyBentOverBall: Bool?
    var followThrough: Bool?

    mutating func updatePoseCriteria(bodyBent: Bool?, followThrough: Bool?) {
        self.bodyBentOverBall = bodyBent
        self.followThrough = followThrough
    }

    // Full classification result from Phase 3 — carries score, badge, criteria lists, and the
    // winning type. UI card (Phase 5) reads from this.
    var classification: ShotClassification?

    mutating func updateClassification(_ classification: ShotClassification) {
        self.classification = classification
    }
}

// MARK: - Pose criteria evaluation
// Given the pose observations captured during a shot, evaluate the two pose-based criteria used
// by the classifier: was the body bent over the ball at contact, and did the kicking leg follow
// through afterward. Each returns Bool? — nil means we couldn't determine (insufficient data or
// low joint confidence at the relevant frames).

struct PoseCriteriaThresholds {
    // Torso must deviate from vertical (in the shot direction) by more than this to count as
    // "bent over the ball". Tuned so natural kicking posture (moderate forward lean) qualifies
    // without dropping the bar so low that just standing upright counts.
    static let bodyBentDegrees: Double = 18
    // Sample this many frames leading up to and including the contact frame when evaluating
    // "Body over ball". Multi-frame sampling makes the criterion robust to Vision noise at any
    // single frame — the player is bent over the ball across a small window (~167ms at 30fps),
    // not just at one exact tick. If ANY frame in the window clears the angle threshold, the
    // criterion fires.
    static let bodyBentWindowFrames: Int = 5
    // Sampling window for follow-through evaluation, in frames after contact. Compare the
    // contact frame's ankle position against every frame in [contact+start, contact+end]. If
    // ANY frame in the window shows the kicking ankle moved past the displacement threshold,
    // the criterion fires. Wider than a single fixed offset — catches both quick snaps
    // (peak displacement ~frame +3) and deliberate strikes (~frame +8) with the same window.
    static let followThroughFrameStart: Int = 3
    static let followThroughFrameEnd: Int = 8
    // Kicking-side ankle must move at least this much (normalized coords) in the shot direction
    // during the follow-through window.
    static let followThroughDisplacement: CGFloat = 0.03
    // Minimum joint confidence to trust a keypoint.
    static let jointConfidence: VNConfidence = 0.1
}

func evaluatePoseCriteria(
    observations: [VNHumanBodyPoseObservation],
    contactFrameIndex: Int,
    shotDirection: ShotDirection
) -> (bodyBentOverBall: Bool?, followThrough: Bool?) {
    guard observations.count >= 3,
          contactFrameIndex >= 0,
          contactFrameIndex < observations.count else {
        return (nil, nil)
    }

    let bent = computeBodyBent(
        observations: observations,
        contactFrameIndex: contactFrameIndex,
        windowFrames: PoseCriteriaThresholds.bodyBentWindowFrames,
        shotDirection: shotDirection
    )
    let follow = computeFollowThrough(
        observations: observations,
        contactFrameIndex: contactFrameIndex,
        windowStart: PoseCriteriaThresholds.followThroughFrameStart,
        windowEnd: PoseCriteriaThresholds.followThroughFrameEnd,
        shotDirection: shotDirection
    )

    return (bent, follow)
}

/// Evaluate the "Body over ball" criterion over a window of frames ending at contact. Iterate
/// backwards from the contact frame across `windowFrames` samples, and return true if ANY frame
/// in the window clears the bend threshold. Return nil only if EVERY sampled frame had pose data
/// too noisy to trust — that way a single unreliable frame doesn't kill the criterion.
private func computeBodyBent(
    observations: [VNHumanBodyPoseObservation],
    contactFrameIndex: Int,
    windowFrames: Int,
    shotDirection: ShotDirection
) -> Bool? {
    let firstIndex = max(0, contactFrameIndex - (windowFrames - 1))
    var anyFrameReliable = false
    for i in firstIndex...contactFrameIndex {
        guard let cleared = computeBodyBentSingleFrame(
            observation: observations[i],
            shotDirection: shotDirection
        ) else {
            // Pose confidence too low at this frame — skip and try the next.
            continue
        }
        anyFrameReliable = true
        if cleared {
            return true
        }
    }
    // Every frame we sampled had usable pose data but none exceeded threshold → definitively false.
    // If NO frame had usable pose data, we couldn't determine — return nil.
    return anyFrameReliable ? false : nil
}

private func computeBodyBentSingleFrame(
    observation: VNHumanBodyPoseObservation,
    shotDirection: ShotDirection
) -> Bool? {
    guard let points = try? observation.recognizedPoints(.all) else { return nil }
    guard let ls = points[.leftShoulder], ls.confidence > PoseCriteriaThresholds.jointConfidence,
          let rs = points[.rightShoulder], rs.confidence > PoseCriteriaThresholds.jointConfidence,
          let lh = points[.leftHip], lh.confidence > PoseCriteriaThresholds.jointConfidence,
          let rh = points[.rightHip], rh.confidence > PoseCriteriaThresholds.jointConfidence else {
        return nil
    }
    // Midpoints in normalized Vision coords (y-up, x-right).
    let shoulderMid = CGPoint(x: (ls.location.x + rs.location.x) / 2,
                              y: (ls.location.y + rs.location.y) / 2)
    let hipMid = CGPoint(x: (lh.location.x + rh.location.x) / 2,
                         y: (lh.location.y + rh.location.y) / 2)
    let torso = CGPoint(x: shoulderMid.x - hipMid.x, y: shoulderMid.y - hipMid.y)

    // Torso must be upright (shoulders above hips) and leaning forward in the shot direction.
    guard torso.y > 0 else { return false }
    let forwardTilt: CGFloat
    switch shotDirection {
    case .right: forwardTilt = torso.x   // shoulders lean toward the goal on the right
    case .left:  forwardTilt = -torso.x
    }
    guard forwardTilt > 0 else { return false }

    let angleDegrees = Double(atan2(forwardTilt, torso.y)) * 180.0 / .pi
    return angleDegrees > PoseCriteriaThresholds.bodyBentDegrees
}

/// Evaluate "Kick follows through" over a window of frames after contact. Sample every frame in
/// [contactIdx+start, contactIdx+end], compare each against the contact frame's ankle position,
/// and fire if ANY comparison shows the kicking ankle moved past threshold. Returns nil only if
/// EVERY sampled comparison had pose data too noisy to trust — a single unreliable frame won't
/// kill the criterion. Mirrors the body-bent multi-frame pattern.
private func computeFollowThrough(
    observations: [VNHumanBodyPoseObservation],
    contactFrameIndex: Int,
    windowStart: Int,
    windowEnd: Int,
    shotDirection: ShotDirection
) -> Bool? {
    let contactObs = observations[contactFrameIndex]
    let firstAfter = contactFrameIndex + windowStart
    let lastAfter = min(contactFrameIndex + windowEnd, observations.count - 1)
    guard firstAfter <= lastAfter else { return nil }

    var anyFrameReliable = false
    for i in firstAfter...lastAfter {
        guard let cleared = computeFollowThroughAcrossPair(
            before: contactObs,
            after: observations[i],
            shotDirection: shotDirection
        ) else {
            // Pose confidence too low at this after-frame — skip and try the next.
            continue
        }
        anyFrameReliable = true
        if cleared {
            return true
        }
    }
    return anyFrameReliable ? false : nil
}

private func computeFollowThroughAcrossPair(
    before: VNHumanBodyPoseObservation,
    after: VNHumanBodyPoseObservation,
    shotDirection: ShotDirection
) -> Bool? {
    guard let beforePoints = try? before.recognizedPoints(.all),
          let afterPoints = try? after.recognizedPoints(.all) else {
        return nil
    }
    let leftDelta = ankleDeltaInShotDirection(
        before: beforePoints[.leftAnkle], after: afterPoints[.leftAnkle], shotDirection: shotDirection)
    let rightDelta = ankleDeltaInShotDirection(
        before: beforePoints[.rightAnkle], after: afterPoints[.rightAnkle], shotDirection: shotDirection)

    // Pick whichever ankle we can measure and moved more in the shot direction — that's the
    // kicking leg (the plant leg barely moves).
    let candidates = [leftDelta, rightDelta].compactMap { $0 }
    guard let bestDelta = candidates.max() else { return nil }
    return bestDelta > PoseCriteriaThresholds.followThroughDisplacement
}

private func ankleDeltaInShotDirection(
    before: VNRecognizedPoint?,
    after: VNRecognizedPoint?,
    shotDirection: ShotDirection
) -> CGFloat? {
    guard let b = before, b.confidence > PoseCriteriaThresholds.jointConfidence,
          let a = after, a.confidence > PoseCriteriaThresholds.jointConfidence else {
        return nil
    }
    let dx = a.location.x - b.location.x
    switch shotDirection {
    case .right: return dx
    case .left:  return -dx
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
    var playerHeightsDuringKick = [CGFloat]()

    // Minimum pose observations required for the kick to be considered "player detected".
    // Below this, the classifier runs on padded zeros and the depth correction is unreliable.
    static let minObservationsForReliableKick = 30

    var wasPlayerDetectedDuringKick: Bool {
        return poseObservations.count >= Self.minObservationsForReliableKick
    }

    var medianPlayerHeightDuringKick: CGFloat? {
        let valid = playerHeightsDuringKick.filter { $0 > 0 }
        guard !valid.isEmpty else { return nil }
        let sorted = valid.sorted()
        return sorted[sorted.count / 2]
    }

    mutating func reset() {
        topSpeed = 0
        avgSpeed = 0
        totalScore = 0
        kickCount = 0
        releaseAngle = 0
        poseObservations = []
        playerHeightsDuringKick = []
    }

    mutating func resetObservations() {
        poseObservations = []
        playerHeightsDuringKick = []
    }

    mutating func storePlayerHeight(_ height: CGFloat) {
        playerHeightsDuringKick.append(height)
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

    mutating func getLastKickType() -> KickType {
        return classifyLastKick().type
    }

    mutating func classifyLastKick() -> (type: KickType, probabilities: [String: Double], observationCount: Int) {
        let count = poseObservations.count
        // Use the cached shared classifier — instantiating ShotClassifier per shot re-loads the
        // whole CoreML model from disk (~100-300ms of dead time between shots).
        guard let actionClassifier = sharedShotClassifier,
              let poseMultiArray = prepareInputWithObservations(poseObservations),
              let predictions = try? actionClassifier.prediction(poses: poseMultiArray) else {
            return (.negative, [:], count)
        }
        let probabilities = predictions.labelProbabilities
        // Only commit to a class when the model is meaningfully confident. Otherwise display ???
        // and let the judgment overlay surface what the model actually leaned toward.
        let topPair = probabilities.max(by: { $0.value < $1.value })
        let type: KickType = {
            guard let pair = topPair,
                  let candidate = KickType(rawValue: pair.key),
                  pair.value >= GameConstants.kickClassifierConfidenceThreshold else {
                return .negative
            }
            return candidate
        }()
        return (type, probabilities, count)
    }
}

struct GameConstants {
    static let maxKicks = 8
    static let newGameTimer = 5
    static let goalLength = 7.32
    static let trajectoryLength = 15
    // Must be at least 120 — the ShotClassifier expects a 120-frame window. With less,
    // prepareInputWithObservations pads the input with zeros and the model defaults toward Negative.
    static let maxPoseObservations = 150
    static let noObservationFrameLimit = 20
    static let maxDistanceWithCurrentTrajectory: CGFloat = 250
    static let maxTrajectoryInFlightPoseObservations = 10
    // If the top class from the action classifier doesn't reach this probability, we display ??? rather
    // than crowning whatever class happened to win by a hair. Tune via test footage.
    static let kickClassifierConfidenceThreshold: Double = 0.5
    // Higher bar for Trivela specifically — Trivela overrides other classifications, so we want
    // the ML model to be really sure before it can overturn a shot that physics/pose said was Laces.
    static let trivelaOverrideConfidenceThreshold: Double = 0.55
    // Instep criterion 1: exit velocity must exceed total velocity by at least this ratio.
    static let instepExitToTotalRatio: Double = 1.3
    // Laces criterion 1: exit velocity floor in MPH. Raised 25 → 35 to give a real separator
    // between laces (raw power) and instep (finesse). A 56mph shot should trip this cleanly.
    static let lacesPowerExitMph: Double = 35
    // Instep criterion 1 (Crisp strike): floor for exit velocity. Also has an upper cap in
    // classifyShot at lacesPowerExitMph so raw-power shots don't get finesse tags.
    static let instepExitMph: Double = 15
    // Instep eligibility gate: UPPER cap for total velocity. Instep is the finesse category —
    // ball loses pace through the curve/dip, so total velocity stays moderate. A wallop with
    // sustained pace ends up in Laces, not here. Formerly the "Controlled pace" criterion —
    // now a required precondition for any instep criterion to fire (not counted).
    static let instepMaxTotalMph: Double = 25
    // Instep criterion 3 (Ball dips): final velocity as a fraction of exit velocity. A ball
    // that decelerates below this ratio by end of trajectory has spin-induced dip or curl —
    // the physical signature of an instep shot. Straight-line laces drives don't decay this
    // hard by end of flight. 0.6 means "ball lost at least 40% of its exit speed by end."
    static let instepBallDipsRatio: Double = 0.6
    // Horizontal field-of-view assumed for the phone's rear wide-angle camera. Modern iPhones sit
    // in the 60–68° range at 1x — 65° is close enough to give player→goal distance within a
    // couple of yards. If we later plumb AVCaptureDevice.activeFormat.videoFieldOfView through
    // to GameManager, use the real value and drop this constant.
    static let assumedHorizontalFOVDegrees: Double = 65
}

// MARK: - Shot classification
// Rules-based classifier that produces the shot's type, score, badge, and per-class criteria list.
// See design doc in-chat for the exact rules — this is where Phase 3 lives.

enum ShotBadge: String {
    case perfect         // 3/3 in a single class
    case trivela         // Trivela ML override triggered (regardless of exit velo)
}

struct ShotClassification {
    var winningType: KickType
    var score: Int
    var badge: ShotBadge?
    var lacesCriteriaMet: [String]
    var instepCriteriaMet: [String]
    var trivelaTriggered: Bool
    var wasPlayerDetected: Bool
    var missedGoal: Bool
    var perfectClass: KickType?  // set to laces or instep when 3/3 achieved
}

/// Run every rule against the shot's measurements and return the final classification.
/// - Parameters:
///   - exitVelocity: MPH just off the foot.
///   - totalVelocity: MPH averaged across the whole flight.
///   - bodyBentOverBall: pose criterion, nil = pose unavailable.
///   - followThrough: pose criterion, nil = pose unavailable.
///   - mlProbabilities: raw class probabilities from ShotClassifier ({"Amazing", "Good", "Great", "Negative"}).
///   - wasPlayerDetected: true if pose observations were sufficient during the kick.
///   - missedGoal: true if the ball's final resting position was outside the goal region.
func classifyShot(
    exitVelocity: Double,
    totalVelocity: Double,
    finalVelocity: Double,
    bodyBentOverBall: Bool?,
    followThrough: Bool?,
    mlProbabilities: [String: Double],
    wasPlayerDetected: Bool,
    missedGoal: Bool
) -> ShotClassification {
    let confidenceThreshold = GameConstants.kickClassifierConfidenceThreshold

    // MARK: Compute Laces criteria
    let lacesMlProb = mlProbabilities[KickType.laces.rawValue] ?? 0
    var lacesCriteriaMet: [String] = []

    if exitVelocity > GameConstants.lacesPowerExitMph {
        lacesCriteriaMet.append("Hit with power")
    }
    if bodyBentOverBall == true {
        lacesCriteriaMet.append("Body over ball")
    }
    if followThrough == true {
        lacesCriteriaMet.append("Kick follows through")
    }
    // Pose fallback: only if BOTH pose criteria are unavailable AND ML is confident, add one
    // ML-derived criterion. This substitutes for the missing pose data without double-counting.
    if bodyBentOverBall == nil && followThrough == nil && lacesMlProb > confidenceThreshold {
        lacesCriteriaMet.append("Technique detected")
    }
    // Cap at 3 — there are only 3 rules for Laces even if we somehow appended more.
    let lacesCount = min(lacesCriteriaMet.count, 3)

    // MARK: Compute Instep criteria
    let instepMlProb = mlProbabilities[KickType.instep.rawValue] ?? 0
    var instepCriteriaMet: [String] = []

    // Instep eligibility gate: modest total velocity + below the laces power threshold. This
    // is the instep signature — ball leaves modest, total pace stays low because the shot is
    // a curl/dip, not a drive. Used as a precondition rather than a counted criterion so shots
    // that don't match the instep signature can't accumulate any instep criteria at all.
    let isInstepEligible = totalVelocity < GameConstants.instepMaxTotalMph
                        && exitVelocity < GameConstants.lacesPowerExitMph
    let instepRatioMet = exitVelocity > GameConstants.instepExitMph
                      && totalVelocity > 0
                      && (exitVelocity / totalVelocity) > GameConstants.instepExitToTotalRatio
    if instepRatioMet && isInstepEligible {
        instepCriteriaMet.append("Crisp strike")
    }
    // "Body opens up" is ML-derived. Skip when the laces "Body over ball" pose criterion
    // already fired — they measure overlapping body positioning at contact and can double-tag.
    if instepMlProb > confidenceThreshold
        && !lacesCriteriaMet.contains("Body over ball")
        && isInstepEligible {
        instepCriteriaMet.append("Body opens up")
    }
    // "Ball dips" — final velocity is a small fraction of exit velocity, indicating spin/curve-
    // induced deceleration by end of trajectory. Straight-line power drives don't decay this
    // hard. exit > 0 guards a divide-by-zero on trajectories that never got real data.
    if exitVelocity > 0
        && (finalVelocity / exitVelocity) < GameConstants.instepBallDipsRatio
        && isInstepEligible {
        instepCriteriaMet.append("Ball dips")
    }
    let instepCount = min(instepCriteriaMet.count, 3)

    // MARK: Compute Trivela
    let trivelaMlProb = mlProbabilities[KickType.trivela.rawValue] ?? 0
    let trivelaTriggered = trivelaMlProb > GameConstants.trivelaOverrideConfidenceThreshold

    // MARK: Determine winner + badge (before missed-goal cap)
    // Threshold is 1 unconditionally: any single criterion (velocity, pose, or ML) classifies.
    // Priority is classify-more-often, accept some occasional misclassification — a 56mph laces
    // where only "Hit with power" fired should still be a laces. Perfect badge requires 3/3.
    let classThreshold = 1
    let perfectLaces = lacesCount >= 3
    let perfectInstep = instepCount >= 3
    let perfectClass: KickType? = perfectLaces ? .laces : (perfectInstep ? .instep : nil)

    var winningType: KickType = .negative
    var badge: ShotBadge?

    if let perfect = perfectClass {
        // Priority 1: perfect shot always wins, even over Trivela override.
        winningType = perfect
        badge = .perfect
    } else if trivelaTriggered {
        // Priority 2: Trivela ML override.
        winningType = .trivela
        badge = .trivela
    } else if lacesCount >= classThreshold && lacesCount >= instepCount {
        // Priority 3: highest criteria count. Laces wins ties per user's priority tier.
        winningType = .laces
    } else if instepCount >= classThreshold {
        winningType = .instep
    }
    // else: winningType stays .negative (default)

    // MARK: Determine score
    var score: Int
    if badge == .perfect {
        score = 15
    } else if trivelaTriggered {
        // Trivela alone: 5. Trivela + high-power exit (>35 MPH): 15.
        score = (exitVelocity > GameConstants.lacesPowerExitMph) ? 15 : 5
    } else if lacesCount >= 2 || instepCount >= 2 {
        score = 5
    } else if winningType == .laces && lacesCriteriaMet.contains("Hit with power") {
        // Solo-criterion power shot: a real 35+ mph strike with no reliable pose data still
        // deserves 5, not 1. Without this floor a well-hit wallop scored the same as a shot
        // with a single lucky pose match, because the tally is criterion-count blind to
        // quality. Only fires when Laces is the winning type — instep can't claim this floor.
        score = 5
    } else {
        // 2 criteria total across classes = 3, 1 = 1, 0 = 0.
        let totalMet = lacesCount + instepCount
        switch totalMet {
        case 2: score = 3
        case 1: score = 1
        default: score = 0
        }
    }

    // MARK: Cap for missed shots
    if missedGoal {
        score = min(score, 3)
        badge = nil  // hide badge when the shot missed the goal
    }

    return ShotClassification(
        winningType: winningType,
        score: score,
        badge: badge,
        lacesCriteriaMet: lacesCriteriaMet,
        instepCriteriaMet: instepCriteriaMet,
        trivelaTriggered: trivelaTriggered,
        wasPlayerDetected: wasPlayerDetected,
        missedGoal: missedGoal,
        perfectClass: perfectClass
    )
}

let jointsOfInterest: [VNHumanBodyPoseObservation.JointName] = [
    .rightAnkle,
    .rightKnee,
    .rightHip,
    .leftAnkle,
    .leftKnee,
    .leftHip
]

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

// MARK: - Cached CoreML models
// Loading a CoreML model from disk is expensive (100-300ms). We instantiate ShotClassifier
// once at first use and reuse the same instance for every classification. Prediction is
// thread-safe on CoreML models, so this is safe.
let sharedShotClassifier: ShotClassifier? = {
    return try? ShotClassifier(configuration: MLModelConfiguration())
}()

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
    // Force the shared ShotClassifier to instantiate now so the first shot's classification
    // doesn't pay the model-load cost.
    _ = sharedShotClassifier
}

// MARK: - Activity Classification Helpers

func prepareInputWithObservations(_ observations: [VNHumanBodyPoseObservation]) -> MLMultiArray? {
    let numAvailableFrames = observations.count
    let observationsNeeded = 120
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

// MARK: - Formatting helpers
/// Formats a speed value (always stored in MPH internally) for display, respecting the user's
/// unit preference. Returns a string with the unit suffix.
func formatSpeed(_ mph: Double) -> String {
    if SettingsStore.shared.useMetricUnits {
        let kph = mph * 1.60934
        return String(format: "%.2f KPH", kph)
    }
    return String(format: "%.2f MPH", mph)
}

/// Formats a session date for user display. Format flips between US (MM-dd) and UK (dd-MM)
/// based on the Units toggle so international users see the day-first order they expect.
/// `short: true` uses 2-digit year for tight layouts (e.g., Home's "Last Session" line);
/// `false` uses 4-digit year (e.g., Recordings list rows).
func formatSessionDate(_ date: Date, short: Bool) -> String {
    let df = DateFormatter()
    let year = short ? "yy" : "yyyy"
    df.dateFormat = SettingsStore.shared.useMetricUnits ? "dd-MM-\(year)" : "MM-dd-\(year)"
    return df.string(from: date)
}

/// Sport term that respects the Units toggle. US → "soccer", UK/metric → "football".
func sportTerm() -> String {
    return SettingsStore.shared.useMetricUnits ? "football" : "soccer"
}

/// Formats a distance value (always stored in meters internally) for display, respecting the
/// user's unit preference. One-decimal precision — the underlying calc has ~1m error bounds
/// anyway so more precision would be false confidence.
func formatDistance(_ meters: Double) -> String {
    if SettingsStore.shared.useMetricUnits {
        return String(format: "%.1fm", meters)
    }
    let yards = meters * 1.0936133
    return String(format: "%.1fyds", yards)
}

// MARK: - SettingsStore
// User preferences persisted to UserDefaults. Views observe changes via NotificationCenter and
// re-read the value they care about — no polling, no manual wiring per view.
final class SettingsStore {
    static let shared = SettingsStore()

    static let developerModeDidChange = Notification.Name("SettingsStore.developerModeDidChange")
    static let unitsDidChange = Notification.Name("SettingsStore.unitsDidChange")
    static let showExtraStatsDidChange = Notification.Name("SettingsStore.showExtraStatsDidChange")

    private let defaults = UserDefaults.standard
    private let developerModeKey = "settings.developerMode"
    private let useMetricUnitsKey = "settings.useMetricUnits"
    private let showExtraStatsKey = "settings.showExtraStats"

    private init() {}

    /// When true, the gameplay screen reveals debug UI: classifier probabilities, pose overlays,
    /// per-shot KPI readouts. Default false so regular users see the clean gameplay UI.
    var developerMode: Bool {
        get { defaults.bool(forKey: developerModeKey) }
        set {
            defaults.set(newValue, forKey: developerModeKey)
            NotificationCenter.default.post(name: Self.developerModeDidChange, object: nil)
        }
    }

    /// When true, speeds display in KPH instead of MPH across the entire app. Default false (MPH).
    var useMetricUnits: Bool {
        get { defaults.bool(forKey: useMetricUnitsKey) }
        set {
            defaults.set(newValue, forKey: useMetricUnitsKey)
            NotificationCenter.default.post(name: Self.unitsDidChange, object: nil)
        }
    }

    /// When true, shows an extra per-shot stats box during gameplay (bottom-right corner).
    /// Independent of Developer Mode — for users who want richer feedback without the debug overlay.
    /// Default false to keep the base gameplay UI minimal.
    var showExtraStats: Bool {
        get { defaults.bool(forKey: showExtraStatsKey) }
        set {
            defaults.set(newValue, forKey: showExtraStatsKey)
            NotificationCenter.default.post(name: Self.showExtraStatsDidChange, object: nil)
        }
    }
}
