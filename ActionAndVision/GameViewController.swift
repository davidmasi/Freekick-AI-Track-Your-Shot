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
    private var shotDirection: ShotDirection = .right
    private let trajectoryView = TrajectoryView()
    private let playerBoundingBox = BoundingBoxView()
    private let jointSegmentView = JointSegmentView()
    private var noObservationFrameCount = 0
    private var trajectoryInFlightPoseObservations = 0
    private var consecutiveLostPoseFrames = 0
    private let maxLostPoseFrames = 8  // ~0.27s at 30fps before declaring "lost"
    private var showSummaryGesture: UITapGestureRecognizer!
    private let kickJudgmentLabel = UILabel()
    private let poseTrackerLabel = UILabel()
    // Big flash of the score asset (Score0/1/3/5/15) that appears under the score label whenever
    // a shot completes. Fades in with the rest of the KPI window and fades out with it.
    private let scoreFlashImageView = UIImageView()
    // Bottom-left classification readout: kick type + optional italic badge + bulleted criteria.
    // 0–4 lines depending on the classification result. Same visual family as the pose tracker.
    private let criteriaBoxLabel = UILabel()
    // Bottom-right extra-stats readout, gated on SettingsStore.shared.showExtraStats. Four lines:
    // Shot Angle, Time to Goal, Distance, Final Speed. Independent from Developer Mode.
    private let extraStatsLabel = UILabel()
    // Tracks the current pending clearKPILabels() call so it can be cancelled if a subsequent
    // shot arrives inside the 5s window. Without this, an older shot's timer keeps firing and
    // hides the newer shot's KPI early (looked like a "gauge fills fast and vanishes" glitch).
    private var pendingKPIClearWork: DispatchWorkItem?
    // UUID of the trajectory we're currently tracking (from Vision). Latched when we first
    // start accepting a trajectory; used to filter out other trajectories in the same frame
    // (shadow, second-shot mid-processing). Reset on shot commit or rejection.
    private var currentTrajectoryUUID: UUID?
    private let trajectoryQueue = DispatchQueue(label: "com.ActionAndVision.trajectory", qos: .userInteractive)
    private let bodyPoseDetectionMinConfidence: VNConfidence = 0.6
    private let trajectoryDetectionMinConfidence: VNConfidence = 0.9

    // Shot-validation thresholds. A tracked trajectory must clear ALL of these to count as a real
    // shot. Anything under gets silently discarded — no score, no kick count, no card. Trajectory
    // noise (head bobs, brief flickers, etc.) fails on one of these because it doesn't travel far
    // enough, doesn't last long enough, or resolves to an implausibly low speed.
    private let minShotDistanceMeters: Double = 1.5
    private let minShotDurationSeconds: Double = 0.25
    // Launch-angle window in degrees. Negative = ball starts angled DOWNWARD off the foot,
    // which no real free-kick does; > 70 = near-vertical, not a shot toward the goal. Filters
    // out fodder trajectories from things like shirt/hand motion that happen to fit a parabola.
    private let minShotLaunchAngleDegrees: Double = 0
    private let maxShotLaunchAngleDegrees: Double = 70
    // Validates on the PREVIEW velocity (pre depth-correction), so displayed final MPH after
    // depth correction is typically lower — this 25 roughly filters out anything displaying
    // below ~15–20 mph in practice, depending on the player/goal depth ratio for the session.
    private let minShotExitVelocityMph: Double = 25.0
    // Secondary display-side floor. The preview floor above validates on the pre-depth-correction
    // magnitude; depth correction can then scale the displayed number down significantly when the
    // ball's flight plane sits closer to the camera than the goal plane. Without this second
    // guard, shots pass validation and still display as ~10 mph — not credible on screen. 17 mph
    // is the lowest the user considers a real shot; anything below drops through the same reject
    // path as a preview failure.
    private let minDisplayedExitVelocityMph: Double = 17.0
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
    
    func getScoreLabelAttributedStringForScore(_ score: Int) -> NSAttributedString {
        // 50% gray — the original color from before my Pass 2 restyle. Matches the muted look
        // of the pre-game status text ("Goal detected", "Waiting for player").
        let attrs: [NSAttributedString.Key: Any] = [
            .foregroundColor: UIColor(red: 0.501960814, green: 0.501960814, blue: 0.501960814, alpha: 1)
        ]
        let totalScore = NSMutableAttributedString(string: " Score: ", attributes: attrs)
        totalScore.append(NSAttributedString(string: "\(score) ", attributes: attrs))
        return totalScore
    }
    
    func setUIElements() {
        resetKPILabels()
        playerBoundingBox.borderColor = #colorLiteral(red: 1, green: 1, blue: 1, alpha: 1)
        playerBoundingBox.backgroundOpacity = 0
        playerBoundingBox.isHidden = true
        // These views are pure visualization. Letting them swallow touches blocks the double-tap gesture.
        playerBoundingBox.isUserInteractionEnabled = false
        jointSegmentView.isUserInteractionEnabled = false
        trajectoryView.isUserInteractionEnabled = false
        view.addSubview(playerBoundingBox)
        view.addSubview(jointSegmentView)
        view.addSubview(trajectoryView)
        gameStatusLabel.text = "Waiting for player"
        // Set kick type counters
        goodKickView.throwType = .laces
        greatKickView.throwType = .instep
        amazingKickView.throwType = .trivela

        // Restyle the storyboard-provided score label to match the pose-tracker aesthetic
        // (monospace, white, translucent black background, rounded corners). Position/anchor stays
        // whatever the storyboard set.
        // Font left to the xib (BanglaMN-Bold 17, matching gameStatusLabel / "Goal detected").
        // Text color comes from the attributed string in getScoreLabelAttributedStringForScore
        // (the original 50% gray), so no textColor override needed here either.
        scoreLabel.backgroundColor = .clear   // was translucent black — now no pill
        scoreLabel.textAlignment = .center    // text centered within the label bounds; the label
                                              // itself is already centerX-anchored to safeArea via the xib
        scoreLabel.layer.cornerRadius = 0
        scoreLabel.layer.masksToBounds = false
        scoreLabel.attributedText = getScoreLabelAttributedStringForScore(0)

        for label in [kickJudgmentLabel, poseTrackerLabel] {
            label.translatesAutoresizingMaskIntoConstraints = false
            label.numberOfLines = 0
            label.font = UIFont.monospacedSystemFont(ofSize: 11, weight: .regular)
            label.textColor = .white
            label.backgroundColor = UIColor.black.withAlphaComponent(0.45)
            label.textAlignment = .left
            label.layer.cornerRadius = 4
            label.layer.masksToBounds = true
            label.isHidden = true
            view.addSubview(label)
        }
        // Score-asset flash — sits directly under the score label in the top-right.
        scoreFlashImageView.translatesAutoresizingMaskIntoConstraints = false
        scoreFlashImageView.contentMode = .scaleAspectFit
        scoreFlashImageView.alpha = 0
        scoreFlashImageView.isUserInteractionEnabled = false
        view.addSubview(scoreFlashImageView)

        // Bottom-left classification readout — same aesthetic as the pose tracker: translucent
        // black background, monospace text, 4pt rounded corners. Hidden until a shot completes.
        criteriaBoxLabel.translatesAutoresizingMaskIntoConstraints = false
        criteriaBoxLabel.numberOfLines = 0
        criteriaBoxLabel.textColor = .white
        criteriaBoxLabel.backgroundColor = UIColor.black.withAlphaComponent(0.45)
        criteriaBoxLabel.textAlignment = .left
        criteriaBoxLabel.layer.cornerRadius = 4
        criteriaBoxLabel.layer.masksToBounds = true
        criteriaBoxLabel.isHidden = true
        view.addSubview(criteriaBoxLabel)

        // Extra stats box — same aesthetic as the criteria box, bottom-right mirror.
        extraStatsLabel.translatesAutoresizingMaskIntoConstraints = false
        extraStatsLabel.numberOfLines = 3
        extraStatsLabel.textColor = .white
        extraStatsLabel.backgroundColor = UIColor.black.withAlphaComponent(0.45)
        extraStatsLabel.textAlignment = .left
        extraStatsLabel.font = UIFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        extraStatsLabel.layer.cornerRadius = 4
        extraStatsLabel.layer.masksToBounds = true
        extraStatsLabel.isHidden = true
        view.addSubview(extraStatsLabel)

        // Default-shot image view — used when the classifier returns .negative. Overlays
        // gameStatusLabel's slot so the pop reads in the same visual position users are used to.
        // Tinted white via .alwaysTemplate on the image itself (see the negative-branch block in
        // the KickCompletedState handler). Finesse/instep asset has different intrinsic proportions
        // than the other three, so if we ever route non-negative types through here we'd need to
        // size it differently — for now it's negative-only and 70pt is a clean fit.

        // Bottom-right per-shot stats — same style as criteria box, three fixed lines.

        NSLayoutConstraint.activate([
            // Bottom-left criteria readout — pinned to the safe-area corner. This is the primary
            // user-facing box in that slot; the dev-mode judgment (below) stacks above it.
            criteriaBoxLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            criteriaBoxLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -12),
            criteriaBoxLabel.widthAnchor.constraint(lessThanOrEqualToConstant: 220),
            // Extra stats box mirrors criteria box on the opposite corner.
            extraStatsLabel.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            extraStatsLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -12),
            extraStatsLabel.widthAnchor.constraint(lessThanOrEqualToConstant: 200),
            // Dev-mode classifier judgment: top-right corner. Wider max so the 3-line output has
            // room without wrapping — the previous 180pt cap was where the format broke down.
            kickJudgmentLabel.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            kickJudgmentLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            kickJudgmentLabel.widthAnchor.constraint(lessThanOrEqualToConstant: 200),
            // Live pose tracker: top-left
            poseTrackerLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            poseTrackerLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            poseTrackerLabel.widthAnchor.constraint(lessThanOrEqualToConstant: 220),
            // Score flash: pinned to the score label's trailing edge, 100pt square just below it.
            // Score circle takes over the scoreLabel's spot during the shot-UI window.
            // centerX matches the label; top hugs the safe area so the circle doesn't spill
            // into the notch/status bar area (scoreLabel is anchored at safeArea.top - 7.5,
            // which is a text-only trick that doesn't work for a taller circle).
            scoreFlashImageView.centerXAnchor.constraint(equalTo: scoreLabel.centerXAnchor),
            scoreFlashImageView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 8),
            scoreFlashImageView.widthAnchor.constraint(equalToConstant: 70),
            scoreFlashImageView.heightAnchor.constraint(equalToConstant: 70)
        ])
    }
    
    func resetKPILabels() {
        // Reset Speed and kickType image
        dashboardView.speed = 0
        kickTypeImage.image = nil
        kickTypeImage.layer.removeAllAnimations()   // stop any in-flight spring pop
        kickTypeImage.transform = .identity   // clear any per-type scale from a prior shot
        kickTypeImage.alpha = 1               // reset alpha (pop animation zeros it before ramp)
        // Cancel any pending KPI clear — a new shot supersedes the previous shot's clear timer.
        pendingKPIClearWork?.cancel()
        pendingKPIClearWork = nil
        // Hide KPI labels
        dashboardView.isHidden = true
        speedStackView.isHidden = true
        metricsStackView.isHidden = true
        // Kill any residual score flash from a prior shot without animating.
        scoreFlashImageView.layer.removeAllAnimations()
        scoreFlashImageView.alpha = 0
        scoreFlashImageView.image = nil
        // Reset the classification readout to its empty state.
        criteriaBoxLabel.isHidden = true
        criteriaBoxLabel.attributedText = nil
        // Also clear the extra-stats box between shots.
        extraStatsLabel.isHidden = true
        extraStatsLabel.text = nil
    }

    func updateKPILabels() {
        // Middle-bottom of screen: gauge animation with speedStackView (speed number + unit label)
        // pinned by the xib to sit directly under it. That's the intended "speed beneath the gauge"
        // display — do NOT add a separate programmatic speed label here.
        dashboardView.isHidden = false
        speedStackView.isHidden = false
        // The xib bakes two subviews into speedStackView: the numeric label (`speedLabel` IBOutlet)
        // and a static "MPH" units label. Convert the internally-MPH top speed if metric, and
        // drive both the number and the unit label so they stay in sync with the settings toggle.
        // Uses topSpeed = max(exit, total) so shots where the 5-point exit sample missed the
        // peak but total captured it read at the higher, truer number.
        let metric = SettingsStore.shared.useMetricUnits
        let displayValue = metric ? lastKickMetrics.topSpeed * 1.60934 : lastKickMetrics.topSpeed
        speedLabel.text = String(format: "%.1f", displayValue)
        if let unitsLabel = speedStackView.arrangedSubviews.compactMap({ $0 as? UILabel }).last, unitsLabel !== speedLabel {
            unitsLabel.text = metric ? "KPH" : "MPH"
        }

        // Restored to user-facing (was gated behind dev mode). Small kick icon overlaid on the
        // gauge + the side metrics cluster (kick type text + launch angle + per-shot score) —
        // these are the original "action center" the app was designed around.
        kickTypeImage.isHidden = false
        metricsStackView.isHidden = false
        kickTypeLabel.text = lastKickMetrics.kickType.displayText
        releaseAngleLabel.text = "\(lastKickMetrics.releaseAngle)°"
        // Was lastKickMetricsLabel.text = "+\(lastKickMetrics.score.rawValue)"
        lastKickMetricsLabel.text = "  "
        // Update the running total on scoreLabel FIRST — the label is about to be hidden for
        // the 5s shot window; when it reappears via clearKPILabels, its text already reflects
        // this shot's points. No stale "old total" flash on the way out or in.
        scoreLabel.attributedText = getScoreLabelAttributedStringForScore(gameManager.playerStats.totalScore)
        scoreLabel.isHidden = true
        // Update kick type image
        kickTypeImage.image = UIImage(named: lastKickMetrics.kickType.imageName)?
            .withRenderingMode(.alwaysTemplate)
        kickTypeImage.tintColor = .white
        // Per-type "rest" transforms to normalize how each icon reads in the 48×38 gauge slot.
        // Tune the constants here directly — reset happens automatically in resetKPILabels.
        //   .instep (Finesse): heavy transparent padding around a small glyph → scale up.
        //   .laces  (Power):   glyph sits slightly right of center in its PNG → nudge left +
        //                      tiny scale-up so it reads at parity with the others.
        //   others: intrinsic size.
        let restTransform: CGAffineTransform
        switch lastKickMetrics.kickType {
        case .instep:
            restTransform = CGAffineTransform(scaleX: 10.8, y: 10.8)
        case .laces:
            restTransform = CGAffineTransform(translationX: -4, y: 0).scaledBy(x: 1.23, y: 1.23)
        default:
            restTransform = .identity
        }

        // Spring pop-in animation. Start at 50% of the per-type rest scale + transparent, spring
        // to the full rest transform + opaque. scaledBy composes cleanly onto any restTransform,
        // so the "pop" is proportional whether the icon's normal size is 1× or 10×. The icon
        // then sits at its rest transform for the remaining KPI window and hides via isHidden.
        kickTypeImage.transform = restTransform.scaledBy(x: 0.5, y: 0.5)
        kickTypeImage.alpha = 0
        UIView.animate(
            withDuration: 0.35,
            delay: 0,
            usingSpringWithDamping: 0.6,
            initialSpringVelocity: 0.5,
            options: [.curveEaseOut],
            animations: {
                self.kickTypeImage.transform = restTransform
                self.kickTypeImage.alpha = 1
            },
            completion: nil
        )
        // Classifier probabilities readout — dev-mode gated (only for debugging).
        kickJudgmentLabel.attributedText = makeJudgmentText(from: lastKickMetrics)
        kickJudgmentLabel.isHidden = !SettingsStore.shared.developerMode

        // User-facing classification box in the bottom-left.
        updateCriteriaBox()
        // Bottom-right extra-stats box — only when the user has opted in via Settings.
        updateExtraStatsBox()
        // Update kick type counters
        switch lastKickMetrics.kickType {
        case .instep:
            greatKickView.incrementKickCount()
        case .laces:
            goodKickView.incrementKickCount()
        case .trivela:
            amazingKickView.incrementKickCount()
        default:
            break
        }
        // Update score for shot views
        let shotView = shots[playerStats.kickCount - 1]
        shotView.image = UIImage(named: "Score\(lastKickMetrics.score.rawValue)")
        // Flash the corresponding score asset under the score label — same shot-UI window.
        scoreFlashImageView.image = UIImage(named: "Score\(lastKickMetrics.score.rawValue)")
        scoreFlashImageView.layer.removeAllAnimations()
        UIView.animate(withDuration: 0.2) { self.scoreFlashImageView.alpha = 1 }
        // Hold the KPI cluster on-screen for 5 seconds so the user has time to read them.
        // Cancel any prior pending clear before scheduling this shot's — otherwise a rapid
        // second shot's KPI can be nuked at the previous shot's original 5s mark.
        pendingKPIClearWork?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.clearKPILabels() }
        pendingKPIClearWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0, execute: work)
    }
    
    func clearKPILabels() {
        dashboardView.isHidden = true
        speedStackView.isHidden = true
        metricsStackView.isHidden = true
        kickTypeImage.isHidden = true
        kickJudgmentLabel.isHidden = true
        criteriaBoxLabel.isHidden = true
        extraStatsLabel.isHidden = true
        // Fade the score flash out alongside the rest of the shot UI, then reveal the running
        // total in scoreLabel again. Text was already updated in updateKPILabels, so the label
        // reappears showing the fresh total — no flicker of the old value.
        UIView.animate(withDuration: 0.25, animations: {
            self.scoreFlashImageView.alpha = 0
        }, completion: { _ in
            self.scoreFlashImageView.image = nil
            self.scoreLabel.isHidden = false
        })
    }

    private func makeJudgmentText(from metrics: KickMetrics) -> NSAttributedString {
        // Three trained kick types in most-common → least-common order. .negative omitted —
        // it means "not any kick type," which isn't useful for calibrating classifier accuracy.
        // No paragraph style, no ▶ marker: the marker character isn't monospace-width in the
        // system monospace font (it renders wider than a space) which was breaking the line
        // width calculation and forcing wraps. Winning row is bolded + accent-green instead.
        let orderedTypes: [KickType] = [.laces, .instep, .trivela]
        let topRaw = metrics.kickProbabilities.max(by: { $0.value < $1.value })?.key
        let monoRegular = UIFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        let monoBold = UIFont.monospacedSystemFont(ofSize: 11, weight: .bold)
        let baseAttrs: [NSAttributedString.Key: Any] = [
            .font: monoRegular,
            .foregroundColor: UIColor.white
        ]
        let winnerAttrs: [NSAttributedString.Key: Any] = [
            .font: monoBold,
            .foregroundColor: HowItWorksViewController.accentGreen
        ]
        let result = NSMutableAttributedString(
            string: " Model judgment\n",
            attributes: [.font: monoBold, .foregroundColor: UIColor.white]
        )
        // Format: " Laces:    50%" — leading space for pill padding, name+colon padded to 9,
        // right-aligned integer percent. Monospace + fixed widths → percentages line up cleanly.
        for type in orderedTypes {
            let prob = metrics.kickProbabilities[type.rawValue] ?? 0
            let isTop = (type.rawValue == topRaw)
            let namePart = "\(type.displayText):".padding(toLength: 9, withPad: " ", startingAt: 0)
            let pct = Int((prob * 100).rounded())
            let line = String(format: " %@%3d%% \n", namePart, pct)
            result.append(NSAttributedString(string: line, attributes: isTop ? winnerAttrs : baseAttrs))
        }
        let reliable = metrics.observationCount >= PlayerStats.minObservationsForReliableKick
        result.append(NSAttributedString(string: " obs: \(metrics.observationCount) \(reliable ? "ok" : "low") ",
                                        attributes: baseAttrs))
        return result
    }
    
    /// Populate the bottom-left criteria box from `lastKickMetrics.classification`.
    /// Line counts per user's spec:
    ///   0 = winningType .negative (no classification) → hidden
    ///   1 = trivela (no criteria to enumerate) → header only
    ///   2 = classified type with 1 criterion → header + 1 bullet
    ///   3 = classified type with 2 criteria → header + 2 bullets
    ///   4 = perfect (laces/instep with 3/3) → header + 3 bullets
    private func updateCriteriaBox() {
        guard let cls = lastKickMetrics.classification, cls.winningType != .negative else {
            criteriaBoxLabel.attributedText = nil
            criteriaBoxLabel.isHidden = true
            return
        }

        let bullets: [String]
        switch cls.winningType {
        case .laces: bullets = cls.lacesCriteriaMet
        case .instep: bullets = cls.instepCriteriaMet
        default: bullets = []   // trivela: no criteria enumerated
        }

        let headerFont = UIFont.monospacedSystemFont(ofSize: 12, weight: .bold)
        let bulletFont = UIFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        let badgeFont = UIFont.italicSystemFont(ofSize: 12)
        let whiteAttrs: [NSAttributedString.Key: Any] = [.font: headerFont, .foregroundColor: UIColor.white]
        let bulletAttrs: [NSAttributedString.Key: Any] = [.font: bulletFont, .foregroundColor: UIColor.white]

        // Leading + trailing whitespace mimics the pose-tracker "padding" trick.
        let result = NSMutableAttributedString(string: " \(cls.winningType.displayText)", attributes: whiteAttrs)

        // Only render an italic badge for .perfect — Trivela's badge is redundant with the type.
        if let badge = cls.badge, badge == .perfect {
            let badgeAttrs: [NSAttributedString.Key: Any] = [
                .font: badgeFont,
                .foregroundColor: HowItWorksViewController.accentGreen
            ]
            result.append(NSAttributedString(string: " Perfect kick! ", attributes: badgeAttrs))
        } else {
            result.append(NSAttributedString(string: " ", attributes: whiteAttrs))
        }

        for bullet in bullets {
            result.append(NSAttributedString(string: "\n * \(bullet) ", attributes: bulletAttrs))
        }

        criteriaBoxLabel.attributedText = result
        criteriaBoxLabel.isHidden = false
    }

    /// Populate the bottom-right extra-stats box (Shot Angle, Time to Goal, Distance).
    /// Gated behind SettingsStore.shared.showExtraStats — off by default. Distance falls back to
    /// the standard 18yd guidance when the FOV calc couldn't be trusted.
    ///
    /// Final Speed was tried here but Vision's last-burst reading is too noisy to trust for
    /// display — the number is often a tiny "cleanup" fragment at the end of the trajectory
    /// rather than the ball's actual arrival speed. `finalVelocity` still drives the Ball Dips
    /// classification criterion where its relative magnitude (not absolute) is what matters.
    private func updateExtraStatsBox() {
        guard SettingsStore.shared.showExtraStats else {
            extraStatsLabel.isHidden = true
            extraStatsLabel.text = nil
            return
        }
        let angle = lastKickMetrics.releaseAngle
        let flight = lastKickMetrics.flightTimeSeconds
        let distanceMeters = lastKickMetrics.distanceToGoalMeters ?? 16.5   // 18 yd default
        let lines = [
            String(format: " Shot Angle:   %.1f° ", angle),
            String(format: " Time to Goal: %.2fs ", flight),
            " Distance:     \(formatDistance(distanceMeters)) "
        ]
        extraStatsLabel.text = lines.joined(separator: "\n")
        extraStatsLabel.isHidden = false
    }

    /// Estimate player→goal distance in meters using similar triangles from the observed goal
    /// pixel width, an assumed camera FOV, and the depth-correction ratio derived from the
    /// player's apparent height. Returns nil when any input is missing or the result falls
    /// outside a plausible band (5–40 m) — callers should fall back to the setup default.
    private func computeDistanceToGoalMeters() -> Double? {
        let goalWidthPoints = gameManager.goalRegion.width
        guard goalWidthPoints > 0 else { return nil }
        let viewWidth = view.bounds.width
        guard viewWidth > 0 else { return nil }

        let fovRad = GameConstants.assumedHorizontalFOVDegrees * .pi / 180
        let focalLengthPx = (viewWidth / 2) / tan(fovRad / 2)
        // Camera → goal distance via similar triangles: f * realGoalWidth / observedGoalWidth.
        let dGoal = focalLengthPx * GameConstants.goalLength / Double(goalWidthPoints)

        // Player → goal only computable when we have a trusted player height. The apparent-size
        // ratio below evaluates to D_player / D_goal (derived from the similar-triangles math:
        // player_pixels/player_meters ∝ 1/D_player, and same for goal). In a normal setup the
        // player is CLOSER to the camera than the goal, so the ratio is < 1.
        guard playerStats.wasPlayerDetectedDuringKick,
              let playerHeight = playerStats.medianPlayerHeightDuringKick,
              playerHeight > 0 else { return nil }

        let averagePlayerHeightMeters = 1.75
        let depthRatio = (averagePlayerHeightMeters * Double(goalWidthPoints))
                       / (GameConstants.goalLength * Double(playerHeight))
        // In a normal setup the ratio is < 1 (player closer than goal). If it's >= 1, the
        // perspective looks wrong — player appears equally distant or farther than the goal —
        // which almost certainly means bad pose data. Bail.
        guard depthRatio < 1.0 else { return nil }

        // D_player = D_goal * (D_player / D_goal) = dGoal * depthRatio.
        let dPlayer = dGoal * depthRatio
        let playerToGoal = dGoal - dPlayer

        // Sanity band: below 5m and the setup guidance is being ignored; above 40m and the
        // detection is almost certainly wrong. Either way, prefer the 18yd default.
        guard playerToGoal >= 5, playerToGoal <= 40 else { return nil }
        return playerToGoal
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
        let poseConfidence = observation.confidence
        DispatchQueue.main.async {
            self.jointSegmentView.joints = joints
            // Pose overlays (joints painted on the player, bounding box around them) are
            // developer-mode only — they clutter the play view for regular users but are
            // essential for calibrating pose tracking during development.
            let showPose = SettingsStore.shared.developerMode
            self.jointSegmentView.alpha = showPose ? 1 : 0
            self.playerBoundingBox.alpha = showPose ? 1 : 0
        }
        // Store the body pose observation in playerStats when the game is in TrackKicksState.
        // We will use these observations for action classification once the throw is complete.
        let inTrackKicks = gameManager.stateMachine.currentState is GameManager.TrackKicksState
        if inTrackKicks {
            playerStats.storeObservation(observation)
            if trajectoryView.inFlight {
                trajectoryInFlightPoseObservations += 1
            }
        }
        updatePoseTrackerLabel(confidence: poseConfidence, observationCount: playerStats.poseObservations.count, isTracking: inTrackKicks, lost: false)
        return box
    }

    private func updatePoseTrackerLabel(confidence: VNConfidence, observationCount: Int, isTracking: Bool, lost: Bool) {
        let bar = String(repeating: "█", count: min(Int(confidence * 10), 10))
            + String(repeating: "·", count: 10 - min(Int(confidence * 10), 10))
        let stateLabel: String
        if lost {
            stateLabel = "LOST"
        } else {
            stateLabel = isTracking ? "tracking" : "idle"
        }
        let reliable = observationCount >= PlayerStats.minObservationsForReliableKick
        let text = String(format: " pose: %@  %.2f\n state: %@\n obs:  %d %@ ",
                          bar, confidence, stateLabel, observationCount, reliable ? "ok" : "low")
        DispatchQueue.main.async {
            self.poseTrackerLabel.text = text
            // Dev-mode only — same rationale as the joint/bounding overlays above.
            self.poseTrackerLabel.isHidden = !SettingsStore.shared.developerMode
        }
    }
    
    // Define regions to filter relevant trajectories for the game
    // kickRegion: Region on the same side of the player as the goal
    // targetRegion: Region around the goal to determine end of kick
    func resetTrajectoryRegions() {
        let goalRegion = gameManager.goalRegion
        let playerRegion = playerBoundingBox.frame
        let kickWindowXBuffer: CGFloat = 50
        let kickWindowYBuffer: CGFloat = 50
        let targetWindowXBuffer: CGFloat = 50
        let kickRegionWidth: CGFloat = 400

        guard !goalRegion.isNull, !playerRegion.isNull else {
            kickRegion = .null
            targetRegion = .null
            shotDirection = .right
            trajectoryView.shotDirection = shotDirection
            return
        }

        shotDirection = goalRegion.midX < playerRegion.midX ? .left : .right
        trajectoryView.shotDirection = shotDirection

        if shotDirection == .right {
            kickRegion = CGRect(x: playerRegion.maxX + kickWindowXBuffer, y: 0, width: kickRegionWidth, height: playerRegion.maxY - kickWindowYBuffer)
        } else {
            kickRegion = CGRect(x: playerRegion.minX - kickRegionWidth - kickWindowXBuffer, y: 0, width: kickRegionWidth, height: playerRegion.maxY - kickWindowYBuffer)
        }

        targetRegion = CGRect(x: goalRegion.minX - targetWindowXBuffer, y: 0,
                              width: goalRegion.width + 2 * targetWindowXBuffer, height: goalRegion.maxY)
    }
    
    // Adjust the kickRegion based on trajectory location.
    // Move the kickRegion toward the target region.
    func updateTrajectoryRegions() {
        let trajectoryLocation = trajectoryView.fullTrajectory.currentPoint
        let didShotCrossCenterOfKickRegion: Bool
        switch shotDirection {
        case .right:
            didShotCrossCenterOfKickRegion = trajectoryLocation.x > kickRegion.origin.x + kickRegion.width / 2
        case .left:
            didShotCrossCenterOfKickRegion = trajectoryLocation.x < kickRegion.origin.x + kickRegion.width / 2
        }
        guard !(kickRegion.contains(trajectoryLocation) && didShotCrossCenterOfKickRegion) else {
            return
        }
        // Overlap buffer window between kickRegion and targetRegion
        let overlapWindowBuffer: CGFloat = 50
        if targetRegion.contains(trajectoryLocation) {
            // When shot is in target region, set the kickRegion to targetRegion.
            kickRegion = targetRegion
        } else {
            switch shotDirection {
            case .right:
                if trajectoryLocation.x + kickRegion.width / 2 - overlapWindowBuffer < targetRegion.origin.x {
                    kickRegion.origin.x = trajectoryLocation.x - kickRegion.width / 2
                }
            case .left:
                if trajectoryLocation.x - kickRegion.width / 2 + overlapWindowBuffer > targetRegion.maxX {
                    kickRegion.origin.x = trajectoryLocation.x - kickRegion.width / 2
                }
            }
        }
        trajectoryView.roi = kickRegion
    }
    
    func processTrajectoryObservations(_ controller: CameraViewController, _ results: [VNTrajectoryObservation]) {
        // Filter to trajectories that clear the confidence bar.
        let qualifying = results.filter { $0.confidence > trajectoryDetectionMinConfidence }

        // Pick ONE trajectory per frame. Two selection modes:
        //   - In-flight: only accept the trajectory whose UUID matches the one we latched at
        //     shot start. This filters out shadows and rapid follow-up shots — both appear as
        //     different UUIDs and get ignored while we're tracking the real shot.
        //   - Not in-flight: pick the most-detected-points trajectory (tie-break by confidence).
        //     Real ball almost always has more sustained parabolic detection than shadows or
        //     incidental motion. Latch that trajectory's UUID for the rest of the shot.
        let chosen: VNTrajectoryObservation?
        if self.trajectoryView.inFlight, let currentUUID = self.currentTrajectoryUUID {
            chosen = qualifying.first(where: { $0.uuid == currentUUID })
        } else {
            chosen = qualifying.max { a, b in
                if a.detectedPoints.count != b.detectedPoints.count {
                    return a.detectedPoints.count < b.detectedPoints.count
                }
                return a.confidence < b.confidence
            }
        }

        guard let path = chosen else {
            // No usable trajectory this frame. If in-flight, Vision has lost our shot's UUID
            // (or the tracked ball moved out of the detected set). Count toward the no-obs
            // limit and commit the shot when it saturates — same lifecycle as before, just
            // additionally triggered when the UUID goes missing rather than only when Vision
            // returns zero trajectories.
            if self.trajectoryView.inFlight {
                self.noObservationFrameCount += 1
                if self.noObservationFrameCount > GameConstants.noObservationFrameLimit {
                    self.updatePlayerStats(controller)
                }
            }
            return
        }

        // Latch the UUID before feeding points into the trajectory view. If this burst gets
        // rejected downstream (ROI or direction check fails inside updatePathLayer), inFlight
        // stays false and we'll re-pick next frame — no harm in over-latching.
        if !self.trajectoryView.inFlight {
            self.currentTrajectoryUUID = path.uuid
        }

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
    
    func updatePlayerStats(_ controller: CameraViewController) {
        let finalShotLocation = trajectoryView.finalShotLocation
        // Snapshot all velocity + angle values BEFORE resetting — resetPath() zeros this state.
        let rawExitVelocity = trajectoryView.exitVelocity
        let rawTotalVelocity = trajectoryView.totalVelocity
        let rawFinalVelocity = trajectoryView.finalVelocity
        let rawLaunchAngle = trajectoryView.launchAngle
        let rawFlightDistancePoints = trajectoryView.totalDistancePoints
        let rawFlightDurationSeconds = trajectoryView.totalDurationSeconds

        // Preview conversions used both for validation and (later) for storing on KickMetrics.
        // Depth correction is skipped for validation because we only need approximate meters here
        // to decide "is this even a shot?" — real per-shot correction happens below on velocities.
        let baseMultiplier = gameManager.pointToMeterMultiplier
        let previewDistanceMeters = Double(rawFlightDistancePoints) * baseMultiplier
        let previewExitMph = rawExitVelocity * baseMultiplier * 2.23694
        let previewTotalMph = rawTotalVelocity * baseMultiplier * 2.23694
        // Gate on max(exit, total) so shots where the 5-point exit sample missed the peak but
        // total captured it aren't rejected as noise. Same floor value, more permissive picker.
        let previewTopMph = max(previewExitMph, previewTotalMph)

        // Validate — must clear ALL floors. Any single failure means this is noise, not a shot.
        let isValidShot = previewDistanceMeters >= minShotDistanceMeters
                       && rawFlightDurationSeconds >= minShotDurationSeconds
                       && previewTopMph >= minShotExitVelocityMph
                       && rawLaunchAngle >= minShotLaunchAngleDegrees
                       && rawLaunchAngle <= maxShotLaunchAngleDegrees

        guard isValidShot else {
            rejectShot()
            return
        }

        // Second gate: project what would actually display after depth correction. The primary
        // preview gate can pass a shot that then depth-corrects to ~10 mph on screen. Compute the
        // correction here (same math as the final velocity computation below — reused via
        // pointsPerSecToMph) and drop the shot if the projected exit reads below the display
        // floor. If the player wasn't detected reliably, depthCorrection stays 1.0 and this
        // reduces to a check against the preview magnitude, so nothing that passed above is
        // rejected here in that case.
        let goalWidthPoints = gameManager.goalRegion.width
        var depthCorrection: Double = 1.0
        if playerStats.wasPlayerDetectedDuringKick,
           let playerHeight = playerStats.medianPlayerHeightDuringKick, playerHeight > 0, goalWidthPoints > 0 {
            let averagePlayerHeightMeters = 1.75
            depthCorrection = (averagePlayerHeightMeters * Double(goalWidthPoints))
                            / (GameConstants.goalLength * Double(playerHeight))
        }
        let pointsPerSecToMph = baseMultiplier * 2.23694 * depthCorrection
        let projectedExitMph = rawExitVelocity * pointsPerSecToMph
        let projectedTotalMph = rawTotalVelocity * pointsPerSecToMph
        // Same max(exit, total) picker as the preview gate above — this is the number that
        // will display, so gate against it directly.
        let projectedTopMph = max(projectedExitMph, projectedTotalMph)

        guard projectedTopMph >= minDisplayedExitVelocityMph else {
            rejectShot()
            return
        }

        playerStats.storePath(self.trajectoryView.fullTrajectory.cgPath)
        trajectoryView.resetPath()
        currentTrajectoryUUID = nil   // release the tracked UUID; next shot latches its own

        // Run the ML classifier to get raw class probabilities (Trivela ML override + Instep "body
        // opens up" criterion + Laces ML fallback all key off these).
        let mlResult = playerStats.classifyLastKick()
        lastKickMetrics.updateClassifierOutput(
            type: mlResult.type,
            probabilities: mlResult.probabilities,
            observationCount: mlResult.observationCount
        )

        // pointsPerSecToMph (and its depthCorrection input) is already computed above for the
        // display-floor gate — reuse it here so the velocity math is single-sourced. See the
        // comment above the display-floor guard for how depth correction is derived.
        let exitMph = round(rawExitVelocity * pointsPerSecToMph * 100) / 100
        let totalMph = round(rawTotalVelocity * pointsPerSecToMph * 100) / 100
        let finalMph = round(rawFinalVelocity * pointsPerSecToMph * 100) / 100
        let releaseAngle = round(rawLaunchAngle * 100) / 100

        // Approximate the contact frame as the last pose observation before the ball started
        // flying. trajectoryInFlightPoseObservations counts poses captured while the trajectory
        // was already in flight, so contactIdx sits at the boundary between "before" and "during".
        let observationCount = playerStats.poseObservations.count
        let contactIdx = observationCount - trajectoryInFlightPoseObservations - 1
        let poseCriteria = evaluatePoseCriteria(
            observations: playerStats.poseObservations,
            contactFrameIndex: contactIdx,
            shotDirection: shotDirection
        )

        // Phase 3: run the rules-based classifier to determine winning type, score, badge, and
        // per-class criteria met. This is the single source of truth for what the shot IS.
        let missedGoal = didShotMissGoal(controller.viewPointForVisionPoint(finalShotLocation))
        let shotClass = classifyShot(
            exitVelocity: exitMph,
            totalVelocity: totalMph,
            finalVelocity: finalMph,
            bodyBentOverBall: poseCriteria.bodyBentOverBall,
            followThrough: poseCriteria.followThrough,
            mlProbabilities: mlResult.probabilities,
            wasPlayerDetected: playerStats.wasPlayerDetectedDuringKick,
            missedGoal: missedGoal
        )

        // Override the ML-only kickType with the rules-based winner. Everything downstream (KPI
        // display, kick counters, summary) uses this final decision.
        lastKickMetrics.kickType = shotClass.winningType

        // Convert score into the existing Scoring enum so downstream stat math still works.
        let scoreEnum = Scoring(rawValue: shotClass.score) ?? .zero
        lastKickMetrics.updateScoreAndAngle(newScore: scoreEnum, angle: releaseAngle)
        lastKickMetrics.updateVelocities(exit: exitMph, total: totalMph, final: finalMph)
        lastKickMetrics.updateFlightTime(rawFlightDurationSeconds)
        lastKickMetrics.updatePoseCriteria(bodyBent: poseCriteria.bodyBentOverBall, followThrough: poseCriteria.followThrough)
        lastKickMetrics.updateClassification(shotClass)
        // Player→goal distance for the bottom-right stats box. nil is fine — the renderer falls
        // back to 18 yards when this can't be estimated confidently.
        lastKickMetrics.updateDistance(computeDistanceToGoalMeters())
        self.gameManager.stateMachine.enter(GameManager.KickCompletedState.self)
    }

    /// Shared cleanup for shots that failed any validation gate. Silently discards: resets the
    /// trajectory view + pose observations, releases the trajectory UUID, and restores the
    /// original kickRegion (updateTrajectoryRegions() shifts it toward the goal as a shot
    /// progresses; if we don't reset, real subsequent shots start outside the region and never
    /// trigger detection). No kick counted, no card shown, next shot starts fresh.
    private func rejectShot() {
        trajectoryView.resetPath()
        playerStats.resetObservations()
        trajectoryInFlightPoseObservations = 0
        noObservationFrameCount = 0
        currentTrajectoryUUID = nil
        resetTrajectoryRegions()
        trajectoryView.roi = kickRegion
    }

    /// True if the ball's final resting position was outside the goal region (with a small height
    /// buffer for bounces). Used by the classifier to cap the score at +3 for missed shots.
    private func didShotMissGoal(_ finalShotLocation: CGPoint) -> Bool {
        let heightBuffer: CGFloat = 100
        let goalRegion = gameManager.goalRegion
        let extendedGoalRegion = CGRect(x: goalRegion.origin.x,
                                        y: goalRegion.origin.y - heightBuffer,
                                        width: goalRegion.width,
                                        height: goalRegion.height + heightBuffer)
        return !extendedGoalRegion.contains(finalShotLocation)
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
            currentTrajectoryUUID = nil   // defensive: ensure the next shot starts UUID-free
        case is GameManager.KickCompletedState:
            dashboardView.speed = lastKickMetrics.topSpeed
            dashboardView.animateSpeedChart()
            playerStats.adjustMetrics(score: lastKickMetrics.score, speed: lastKickMetrics.topSpeed,
                                      releaseAngle: lastKickMetrics.releaseAngle, kickType: lastKickMetrics.kickType)
            playerStats.resetObservations()
            trajectoryInFlightPoseObservations = 0
            self.updateKPILabels()
            
            // Explicit () -> Void so the closure discards stateMachine.enter's Bool return —
            // otherwise callers passing progressToNext into Void-expecting completion handlers
            // trip an unused-result warning on the call site.
            let progressToNext: () -> Void = {
                if self.playerStats.kickCount == GameConstants.maxKicks {
                    self.gameManager.stateMachine.enter(GameManager.ShowSummaryState.self)
                } else {
                    self.gameManager.stateMachine.enter(GameManager.TrackKicksState.self)
                }
            }

            // Kick type is shown via kickTypeImage (small icon on the gauge, persistent for the
            // 8s KPI window). The word popup that used to fire here was redundant with the icon
            // and had its own separate short lifecycle — deleted. A small delay preserves the
            // between-shot rhythm the old popup gave for free.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
                progressToNext()
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
                // Consider a frame "tracked" only when the model returned a result above the confidence threshold.
                let trackedResult = detectPlayerRequest.results?.first.flatMap { result -> VNHumanBodyPoseObservation? in
                    return result.confidence > bodyPoseDetectionMinConfidence ? result : nil
                }
                if let result = trackedResult {
                    consecutiveLostPoseFrames = 0
                    let box = humanBoundingBox(for: result)
                    let boxView = playerBoundingBox
                    DispatchQueue.main.async {
                        let inset: CGFloat = -20.0
                        let viewRect = controller.viewRectForVisionRect(box).insetBy(dx: inset, dy: inset)
                        self.updateBoundingBox(boxView, withRect: viewRect)
                        // Capture player height in view coordinates during kick tracking — used for
                        // depth correction in the speed calculation. Skip insets so we measure the raw player.
                        if self.gameManager.stateMachine.currentState is GameManager.TrackKicksState, !box.isNull {
                            let rawViewRect = controller.viewRectForVisionRect(box)
                            self.playerStats.storePlayerHeight(rawViewRect.height)
                        }
                        if !self.playerDetected && !boxView.isHidden {
                            self.gameStatusLabel.alpha = 0
                            self.resetTrajectoryRegions()
                            self.gameManager.stateMachine.enter(GameManager.DetectedPlayerState.self)
                        }
                    }
                } else {
                    // Model returned nothing usable. Count consecutive failures; once we cross the
                    // threshold, fade out the visualization and surface a "lost" state in the tracker.
                    consecutiveLostPoseFrames += 1
                    if consecutiveLostPoseFrames > maxLostPoseFrames {
                        let obsCount = playerStats.poseObservations.count
                        DispatchQueue.main.async {
                            self.jointSegmentView.alpha = 0
                            self.jointSegmentView.resetView()
                            self.playerBoundingBox.alpha = 0
                            self.updatePoseTrackerLabel(confidence: 0, observationCount: obsCount, isTracking: false, lost: true)
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
            // No kicks recorded yet — there's nothing meaningful to summarize, so just exit.
            if playerStats.kickCount == 0 {
                (parent as? RootViewController)?.exitToMenu()
                return
            }
            if !gameManager.stateMachine.enter(GameManager.ShowSummaryState.self) {
                (parent as? RootViewController)?.exitToMenu()
            }
        }
    }
}

