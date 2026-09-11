/*
See LICENSE folder for this sample’s licensing information.

Abstract:
View that displays a detected trajectory.
*/

import UIKit
import Vision

class TrajectoryView: UIView, AnimatedTransitioning {
    var roi = CGRect.null
    var inFlight = false
    var outOfROIPoints = 0
    var fullTrajectory = UIBezierPath()
    var duration = 0.0
    var launchAngle = 0.0  // degrees from horizontal at first detected ball motion; + = upward
    var shotDirection: ShotDirection = .right
    var points: [VNPoint] = [] {
        didSet {
            if isTrajectoryMovingForward {
                updatePathLayer()
            }
        }
    }

    // How many points at the start of the first burst are used to estimate exit velocity.
    // 5 points ≈ 0.17s at 30fps — long enough to be robust to a single noisy detection, short enough
    // that the measurement approximates "speed just after foot contact" rather than average flight speed.
    private static let exitVelocityPointCount = 5

    // Cumulative measurements across every accepted burst — used for totalVelocity.
    // Made read-visible to callers so shot-validation can inspect them at commit time.
    private(set) var totalDistancePoints: CGFloat = 0
    private(set) var totalDurationSeconds: Double = 0

    // Measurements from only the first N points of the first burst — used for exitVelocity.
    private var exitDistancePoints: CGFloat = 0
    private var exitDurationSeconds: Double = 0

    // Measurements from only the most recent accepted burst — used for finalVelocity.
    private var lastBurstDistance: CGFloat = 0
    private var lastBurstDuration: Double = 0

    /// Speed in points/second measured over the first ~0.17s of flight — best estimate of the
    /// ball's speed just after leaving the foot.
    var exitVelocity: Double {
        guard exitDurationSeconds > 0 else { return 0 }
        return Double(exitDistancePoints) / exitDurationSeconds
    }

    /// Speed in points/second averaged across the entire flight (sum of per-burst distances /
    /// sum of per-burst durations). Represents the shot's overall pace.
    var totalVelocity: Double {
        guard totalDurationSeconds > 0 else { return 0 }
        return Double(totalDistancePoints) / totalDurationSeconds
    }

    /// Speed in points/second during the most recently accepted burst before the throw completed —
    /// best estimate of the ball's speed as it arrives at the goal.
    var finalVelocity: Double {
        guard lastBurstDuration > 0 else { return 0 }
        return Double(lastBurstDistance) / lastBurstDuration
    }

    private let pathLayer = CAShapeLayer()
    private let blurLayer = CAShapeLayer()
    private let shadowLayer = CAShapeLayer()

    private var distanceWithCurrentTrajectory: CGFloat = 0
    private var isTrajectoryMovingForward: Bool {
        guard let firstPoint = points.first, let lastPoint = points.last else {
            return false
        }
        switch shotDirection {
        case .right:
            return lastPoint.location.x > firstPoint.location.x
        case .left:
            return lastPoint.location.x < firstPoint.location.x
        }
    }

    var isThrowComplete: Bool {
        // Mark throw as complete if we don't get any trajectory observations in our roi
        // for consecutive GameConstants.noObservationFrameLimit frames
        if inFlight && outOfROIPoints > GameConstants.noObservationFrameLimit {
            return true
        }
        return false
    }

    var finalShotLocation: CGPoint {
        let ballLocation = fullTrajectory.currentPoint
        let flipVertical = CGAffineTransform.verticalFlip
        let scaleDown = CGAffineTransform(scaleX: (1 / bounds.width), y: (1 / bounds.height))
        return ballLocation.applying(scaleDown).applying(flipVertical)
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupLayer()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupLayer()
    }

    func resetPath() {
        inFlight = false
        outOfROIPoints = 0
        distanceWithCurrentTrajectory = 0
        fullTrajectory.removeAllPoints()
        pathLayer.path = fullTrajectory.cgPath
        blurLayer.path = fullTrajectory.cgPath
        shadowLayer.path = fullTrajectory.cgPath
        // Reset all velocity-tracking state so the next shot starts clean.
        launchAngle = 0
        totalDistancePoints = 0
        totalDurationSeconds = 0
        exitDistancePoints = 0
        exitDurationSeconds = 0
        lastBurstDistance = 0
        lastBurstDuration = 0
    }

    func addPath(_ path: CGPath) {
        fullTrajectory.cgPath = path
        pathLayer.lineWidth = 2
        pathLayer.path = fullTrajectory.cgPath
        shadowLayer.lineWidth = 4
        shadowLayer.path = fullTrajectory.cgPath
    }

    private func setupLayer() {
        shadowLayer.lineWidth = 12.0
        shadowLayer.lineCap = .round
        shadowLayer.fillColor = UIColor.clear.cgColor
        shadowLayer.strokeColor = #colorLiteral(red: 0.9882352941, green: 0.4666666667, blue: 0, alpha: 0.4519210188).cgColor
        layer.addSublayer(shadowLayer)
        blurLayer.lineWidth = 8.0
        blurLayer.lineCap = .round
        blurLayer.fillColor = UIColor.clear.cgColor
        blurLayer.strokeColor = #colorLiteral(red: 0.9960784314, green: 0.737254902, blue: 0, alpha: 0.597468964).cgColor
        layer.addSublayer(blurLayer)
        pathLayer.lineWidth = 4.0
        pathLayer.lineCap = .round
        pathLayer.fillColor = UIColor.clear.cgColor
        pathLayer.strokeColor = #colorLiteral(red: 0.9960784314, green: 0.737254902, blue: 0, alpha: 0.7512574914).cgColor
        layer.addSublayer(pathLayer)
    }
    
    private func updatePathLayer() {
        let trajectory = UIBezierPath()
        guard let startingPoint = points.first else {
            return
        }
        trajectory.move(to: startingPoint.location)
        for point in points.dropFirst() {
            trajectory.addLine(to: point.location)
        }
        let flipVertical = CGAffineTransform.verticalFlip
        trajectory.apply(flipVertical)
        trajectory.apply(CGAffineTransform(scaleX: bounds.width, y: bounds.height))
        let startScaled = startingPoint.location.applying(flipVertical).applying(CGAffineTransform(scaleX: bounds.width, y: bounds.height))
        var distanceWithCurrentTrajectory: CGFloat = 0
        if inFlight {
            distanceWithCurrentTrajectory = startScaled.distance(to: fullTrajectory.currentPoint)
        }
        if (roi.contains(trajectory.currentPoint) || (inFlight && roi.contains(startScaled))) &&
            distanceWithCurrentTrajectory < GameConstants.maxDistanceWithCurrentTrajectory {
            let burstDistance = trajectory.currentPoint.distance(to: startScaled)
            // Junk-burst filter for the very first burst only. Vision occasionally captures a
            // tiny "hook" of ball movement near the player (foot addressing the ball, small
            // wiggles during the run-up) before the actual shot flies. If we let one of those
            // become the "first burst," it anchors exitVelocity to that slow motion and drags
            // the shot's speed reading way down. A real first burst is many times longer than
            // these blips — require a minimum distance before accepting the first burst.
            let minFirstBurstDistance: CGFloat = 100
            if !inFlight && burstDistance < minFirstBurstDistance {
                return   // discard this micro-burst; leave inFlight=false, wait for a real one
            }
            if !inFlight {
                // Launch angle from the ball's initial trajectory direction. Vision coords are y-up,
                // so dy > 0 means upward motion. Using |dx| so left- and right-side shots both yield
                // a positive value for "above horizontal" / negative for "below."
                if let firstNorm = points.first?.location, let lastNorm = points.last?.location {
                    let dx = lastNorm.x - firstNorm.x
                    let dy = lastNorm.y - firstNorm.y
                    launchAngle = atan2(dy, abs(dx)) * 180.0 / .pi
                }
                // Exit velocity: only the first N points of the very first burst. Approximate the
                // duration proportionally since Vision reports one duration for the whole burst.
                let exitN = min(TrajectoryView.exitVelocityPointCount, points.count)
                if exitN >= 2 {
                    let scaleTransform = CGAffineTransform(scaleX: bounds.width, y: bounds.height)
                    let firstScaled = points.first!.location.applying(flipVertical).applying(scaleTransform)
                    let lastEarlyScaled = points[exitN - 1].location.applying(flipVertical).applying(scaleTransform)
                    exitDistancePoints = firstScaled.distance(to: lastEarlyScaled)
                    exitDurationSeconds = duration * Double(exitN - 1) / Double(max(points.count - 1, 1))
                }
                fullTrajectory = trajectory
            } else {
                fullTrajectory.append(trajectory)
            }
            // Running totals — updated on every accepted burst so total and final velocities are
            // always current at any point during flight.
            totalDistancePoints += burstDistance
            totalDurationSeconds += duration
            lastBurstDistance = burstDistance
            lastBurstDuration = duration
            shadowLayer.path = fullTrajectory.cgPath
            blurLayer.path = fullTrajectory.cgPath
            pathLayer.path = fullTrajectory.cgPath
            outOfROIPoints = 0
            inFlight = true
        } else {
            outOfROIPoints += 1
        }
    }
}

