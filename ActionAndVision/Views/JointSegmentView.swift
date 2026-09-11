/*
See LICENSE folder for this sample’s licensing information.

Abstract:
View that displays a joint segment.
*/

import UIKit
import Vision

class JointSegmentView: UIView, AnimatedTransitioning {
    var joints: [VNHumanBodyPoseObservation.JointName: CGPoint] = [:] {
        didSet {
            updatePathLayer()
        }
    }

    private let jointRadius: CGFloat = 3.0
    private let jointLayer = CAShapeLayer()
    private var jointPath = UIBezierPath()

    private let jointSegmentWidth: CGFloat = 2.0
    private let jointSegmentLayer = CAShapeLayer()
    private var jointSegmentPath = UIBezierPath()

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupLayer()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupLayer()
    }

    func resetView() {
        jointLayer.path = nil
        jointSegmentLayer.path = nil
    }

    private func setupLayer() {
        jointSegmentLayer.lineCap = .round
        jointSegmentLayer.lineWidth = jointSegmentWidth
        jointSegmentLayer.fillColor = UIColor.clear.cgColor
        jointSegmentLayer.strokeColor = #colorLiteral(red: 0.6078431373, green: 0.9882352941, blue: 0, alpha: 1).cgColor
        layer.addSublayer(jointSegmentLayer)
        let jointColor = #colorLiteral(red: 1, green: 1, blue: 1, alpha: 1).cgColor
        jointLayer.strokeColor = jointColor
        jointLayer.fillColor = jointColor
        layer.addSublayer(jointLayer)
    }

    private func updatePathLayer() {
        let flipVertical = CGAffineTransform.verticalFlip
        let scaleToBounds = CGAffineTransform(scaleX: bounds.width, y: bounds.height)
        jointPath.removeAllPoints()
        jointSegmentPath.removeAllPoints()
        let legs: [[VNHumanBodyPoseObservation.JointName]] = [
            [.rightAnkle, .rightKnee, .rightHip],
            [.leftAnkle, .leftKnee, .leftHip]
        ]
        for leg in legs {
            var startedLeg = false
            for jointName in leg {
                guard let joint = joints[jointName] else { continue }
                let scaled = joint.applying(flipVertical).applying(scaleToBounds)
                jointPath.append(UIBezierPath(arcCenter: scaled, radius: jointRadius,
                                              startAngle: 0, endAngle: .pi * 2, clockwise: true))
                if !startedLeg {
                    jointSegmentPath.move(to: scaled)
                    startedLeg = true
                } else {
                    jointSegmentPath.addLine(to: scaled)
                }
            }
        }
        jointLayer.path = jointPath.cgPath
        jointSegmentLayer.path = jointSegmentPath.cgPath
    }
}
