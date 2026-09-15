//
//  HomeViewController.swift
//  Freekick
//
//  Landing screen. Layout, top to bottom:
//    - Top bar with utility icons (help ?, settings gear)
//    - Hero: logo + title + subtitle
//    - Primary action: filled "Start Session" button
//      header ("Last Session: MM-dd-yy (N)") + full-width 16:9 thumbnail of the most recent
//      recording. Tap thumbnail to replay. "(N)" is the chronological ordinal of that recording
//      within its calendar day.
//    - Last Session stats footer: Total Shots · Average Speed · Top Speed (all sourced from the
//      most recent session; the all-time top speed lives on the Settings screen).
//

import UIKit
import AVFoundation

class HomeViewController: UIViewController {

    private let titleLabel = GradientLabel()
    private let subtitleLabel = UILabel()
    private let logoImageView = UIImageView(image: UIImage(named: "freekick-white"))

    private let topBar = UIView()
    private let startButton = UIButton(type: .system)

    // Last Session block
    private let lastSessionLabel = UILabel()               // "Last Session: MM-dd-yy (N)"
    private let recordingsSeeAllButton = UIButton(type: .system)
    private let lastSessionThumbnail = UIImageView()
    private let recordingsEmptyLabel = UILabel()

    // Last-session stats footer (sourced from the most recent recording).
    private let statsStackView = UIStackView()
    private let totalShotsColumn = HomeStatColumn(title: "Total Shots")
    private let topSpeedColumn = HomeStatColumn(title: "Top Speed")

    private var recordings: [SessionRecord] = []

    // Date format for the Last Session line is now derived per-render via formatSessionDate() so
    // it can flip between US (MM-dd) and UK (dd-MM) with the Units toggle. No stored formatter.

    override func viewDidLoad() {
        super.viewDidLoad()
        // Hide anything left over from the storyboard scene.
        view.subviews.forEach { $0.isHidden = true }
        setupLayout()
        NotificationCenter.default.addObserver(self, selector: #selector(storeChanged), name: .sessionStoreDidChange, object: nil)
        // Re-render speed labels when the user flips MPH ↔ KPH from Settings.
        NotificationCenter.default.addObserver(self, selector: #selector(storeChanged), name: SettingsStore.unitsDidChange, object: nil)
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        // Home is the root — no back arrow ever.
        navigationController?.setNavigationBarHidden(true, animated: animated)
        // Clear stale state so subsequent fresh Play flows don't auto-forward past SourcePicker.
        // If a video source is currently set, we've been handed one externally (Files app
        // "Open with Freekick") — leave it alone so viewDidAppear can route it into the flow.
        if GameManager.shared.recordedVideoSource == nil {
            GameManager.shared.directToLiveCamera = false
            GameManager.shared.replayingRecordID = nil
        }
        reloadRecordings()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        // If SceneDelegate handed us an incoming external video (Files app "Open with Freekick"),
        // push into SourcePicker — its viewDidAppear will auto-forward to the analysis flow
        // because recordedVideoSource is already set.
        if GameManager.shared.recordedVideoSource != nil {
            pushStoryboardVC(withIdentifier: "SourcePickerViewController")
            return
        }
        animateHero()
    }

    private func setupLayout() {
        // Background
        let background = UIImageView(image: UIImage(named: "appbackground"))
        background.translatesAutoresizingMaskIntoConstraints = false
        background.contentMode = .scaleAspectFill
        background.clipsToBounds = true
        view.addSubview(background)

        // Darkening gradient — heavier at top for text pop, lighter toward the bottom.
        let overlay = GradientOverlayView()
        overlay.translatesAutoresizingMaskIntoConstraints = false
        overlay.topColor = UIColor.black.withAlphaComponent(0.55)
        overlay.bottomColor = UIColor.black.withAlphaComponent(0.25)
        view.addSubview(overlay)

        // Top bar container
        topBar.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(topBar)

        let helpButton = UIButton(type: .system)
        helpButton.translatesAutoresizingMaskIntoConstraints = false
        helpButton.setImage(UIImage(systemName: "questionmark.circle"), for: .normal)
        helpButton.tintColor = .white
        helpButton.contentHorizontalAlignment = .leading
        helpButton.addTarget(self, action: #selector(howItWorksTapped), for: .touchUpInside)
        topBar.addSubview(helpButton)

        let settingsButton = UIButton(type: .system)
        settingsButton.translatesAutoresizingMaskIntoConstraints = false
        settingsButton.setImage(UIImage(systemName: "gearshape"), for: .normal)
        settingsButton.tintColor = .white
        settingsButton.contentHorizontalAlignment = .trailing
        settingsButton.addTarget(self, action: #selector(settingsTapped), for: .touchUpInside)
        topBar.addSubview(settingsButton)

        // Larger tap targets for both icons.
        let iconPointSize = UIImage.SymbolConfiguration(pointSize: 22, weight: .regular)
        helpButton.setPreferredSymbolConfiguration(iconPointSize, forImageIn: .normal)
        settingsButton.setPreferredSymbolConfiguration(iconPointSize, forImageIn: .normal)

        // Hero
        logoImageView.translatesAutoresizingMaskIntoConstraints = false
        logoImageView.contentMode = .scaleAspectFit
        logoImageView.clipsToBounds = true
        logoImageView.layer.cornerRadius = 14
        logoImageView.layer.borderColor = UIColor.white.cgColor
        logoImageView.layer.borderWidth = 2.0
        view.addSubview(logoImageView)

        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        titleLabel.text = "Freekick"
        titleLabel.font = UIFont(name: "Inter-ExtraBold", size: 46) ?? UIFont.systemFont(ofSize: 46, weight: .heavy)
        titleLabel.textAlignment = .center
        titleLabel.gradientColors = [
            UIColor.white.cgColor,
            UIColor(white: 0.55, alpha: 1).cgColor
        ]
        titleLabel.layer.shadowColor = UIColor.black.cgColor
        titleLabel.layer.shadowRadius = 4
        titleLabel.layer.shadowOpacity = 0.35
        titleLabel.layer.shadowOffset = CGSize(width: 0, height: 1)
        view.addSubview(titleLabel)

        subtitleLabel.translatesAutoresizingMaskIntoConstraints = false
        subtitleLabel.text = "Track Shot Speed and Accuracy with\n Real-Time Metrics"
        subtitleLabel.numberOfLines = 0
        subtitleLabel.font = UIFont.systemFont(ofSize: 16, weight: .regular)
        subtitleLabel.textColor = UIColor.white.withAlphaComponent(0.85)
        subtitleLabel.textAlignment = .center
        view.addSubview(subtitleLabel)

        // Primary Start button — filled white, black text.
        configureStartButton()
        view.addSubview(startButton)

        // Last Session header — dynamic ("Last Session: MM-dd-yy (N)")
        lastSessionLabel.translatesAutoresizingMaskIntoConstraints = false
        lastSessionLabel.font = UIFont.monospacedSystemFont(ofSize: 12, weight: .semibold)
        lastSessionLabel.textColor = UIColor.white.withAlphaComponent(0.7)
        lastSessionLabel.setContentHuggingPriority(.required, for: .horizontal)
        view.addSubview(lastSessionLabel)

        recordingsSeeAllButton.translatesAutoresizingMaskIntoConstraints = false
        recordingsSeeAllButton.setTitle("See all →", for: .normal)
        recordingsSeeAllButton.setTitleColor(UIColor.white.withAlphaComponent(0.85), for: .normal)
        recordingsSeeAllButton.titleLabel?.font = UIFont.systemFont(ofSize: 13, weight: .regular)
        recordingsSeeAllButton.addTarget(self, action: #selector(recordingsTapped), for: .touchUpInside)
        view.addSubview(recordingsSeeAllButton)

        // Full-width thumbnail of the most recent recording. Tap → replay through SourcePicker.
        lastSessionThumbnail.translatesAutoresizingMaskIntoConstraints = false
        lastSessionThumbnail.contentMode = .scaleAspectFill
        lastSessionThumbnail.clipsToBounds = true
        lastSessionThumbnail.layer.cornerRadius = 10
        lastSessionThumbnail.backgroundColor = UIColor.white.withAlphaComponent(0.1)
        lastSessionThumbnail.isUserInteractionEnabled = true
        let thumbTap = UITapGestureRecognizer(target: self, action: #selector(lastSessionThumbnailTapped))
        lastSessionThumbnail.addGestureRecognizer(thumbTap)
        view.addSubview(lastSessionThumbnail)

        recordingsEmptyLabel.translatesAutoresizingMaskIntoConstraints = false
        recordingsEmptyLabel.text = "No recordings yet — tap Start to record your first session"
        recordingsEmptyLabel.font = UIFont.italicSystemFont(ofSize: 13)
        recordingsEmptyLabel.textColor = UIColor.white.withAlphaComponent(0.65)
        recordingsEmptyLabel.textAlignment = .center
        recordingsEmptyLabel.numberOfLines = 0
        recordingsEmptyLabel.isHidden = true
        view.addSubview(recordingsEmptyLabel)

        // Last-session stats footer — three evenly-spaced columns pinned above the bottom safe area.
        statsStackView.translatesAutoresizingMaskIntoConstraints = false
        statsStackView.axis = .horizontal
        statsStackView.distribution = .fillEqually
        statsStackView.alignment = .top
        statsStackView.spacing = 8
        statsStackView.addArrangedSubview(topSpeedColumn)
        statsStackView.addArrangedSubview(totalShotsColumn)
        view.addSubview(statsStackView)

        NSLayoutConstraint.activate([
            background.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            background.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            background.topAnchor.constraint(equalTo: view.topAnchor),
            background.bottomAnchor.constraint(equalTo: view.bottomAnchor),

            overlay.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            overlay.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            overlay.topAnchor.constraint(equalTo: view.topAnchor),
            overlay.bottomAnchor.constraint(equalTo: view.bottomAnchor),

            // Top bar
            topBar.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            topBar.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            topBar.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            topBar.heightAnchor.constraint(equalToConstant: 44),

            helpButton.leadingAnchor.constraint(equalTo: topBar.leadingAnchor, constant: 20),
            helpButton.centerYAnchor.constraint(equalTo: topBar.centerYAnchor),
            helpButton.widthAnchor.constraint(equalToConstant: 44),
            helpButton.heightAnchor.constraint(equalToConstant: 44),

            settingsButton.trailingAnchor.constraint(equalTo: topBar.trailingAnchor, constant: -20),
            settingsButton.centerYAnchor.constraint(equalTo: topBar.centerYAnchor),
            settingsButton.widthAnchor.constraint(equalToConstant: 44),
            settingsButton.heightAnchor.constraint(equalToConstant: 44),

            // Hero — anchored near the top of the screen so the block sits high on the page,
            // with generous internal spacing that gives the title room to breathe.
            logoImageView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            logoImageView.topAnchor.constraint(equalTo: topBar.bottomAnchor, constant: 22),
            logoImageView.widthAnchor.constraint(equalToConstant: 80),
            logoImageView.heightAnchor.constraint(equalToConstant: 80),

            titleLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            titleLabel.topAnchor.constraint(equalTo: logoImageView.bottomAnchor, constant: 18),

            subtitleLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            subtitleLabel.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 18),
            subtitleLabel.leadingAnchor.constraint(greaterThanOrEqualTo: view.leadingAnchor, constant: 30),
            subtitleLabel.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor, constant: -30),

            // Primary action
            startButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            startButton.topAnchor.constraint(equalTo: subtitleLabel.bottomAnchor, constant: 28),
            startButton.widthAnchor.constraint(equalToConstant: 280),
            startButton.heightAnchor.constraint(equalToConstant: 64),

            // Bottom-up chain so the whole recordings block sits low on the screen, leaving the
            // sky/skyline background visible between the hero and the block. Header row and stats
            // both align to the thumbnail's edges so the section reads as a single vertical column.

            // Centered thumbnail — narrower than the section so the background stays visible.
            lastSessionThumbnail.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            lastSessionThumbnail.widthAnchor.constraint(equalToConstant: 260),
            lastSessionThumbnail.heightAnchor.constraint(equalTo: lastSessionThumbnail.widthAnchor, multiplier: 9.0 / 16.0),

            // Stats footer pinned near the bottom safe area, sharing the thumbnail's horizontal
            // edges. Below-stats inset (12) and thumbnail→stats gap (22) redistribute the same
            // total budget as before so the thumbnail's absolute position doesn't shift; the stats
            // themselves sit closer to the screen bottom and get more breathing room above.
            statsStackView.leadingAnchor.constraint(equalTo: lastSessionThumbnail.leadingAnchor),
            statsStackView.trailingAnchor.constraint(equalTo: lastSessionThumbnail.trailingAnchor),
            statsStackView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -12),

            lastSessionThumbnail.bottomAnchor.constraint(equalTo: statsStackView.topAnchor, constant: -22),

            // Header row sits directly above the thumbnail, edge-aligned to it.
            lastSessionLabel.leadingAnchor.constraint(equalTo: lastSessionThumbnail.leadingAnchor),
            lastSessionLabel.bottomAnchor.constraint(equalTo: lastSessionThumbnail.topAnchor, constant: -10),
            recordingsSeeAllButton.trailingAnchor.constraint(equalTo: lastSessionThumbnail.trailingAnchor),
            recordingsSeeAllButton.centerYAnchor.constraint(equalTo: lastSessionLabel.centerYAnchor),

            // Empty state sits in the middle of the thumbnail area when there are no recordings.
            recordingsEmptyLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            recordingsEmptyLabel.centerYAnchor.constraint(equalTo: lastSessionThumbnail.centerYAnchor),
            recordingsEmptyLabel.leadingAnchor.constraint(greaterThanOrEqualTo: view.leadingAnchor, constant: 30),
            recordingsEmptyLabel.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor, constant: -30)
        ])

        // Hero fades in on appear.
        [logoImageView, titleLabel, subtitleLabel].forEach { $0.alpha = 0 }
    }

    private func configureStartButton() {
        startButton.translatesAutoresizingMaskIntoConstraints = false
        startButton.setTitle("Start Session", for: .normal)

        // Center the title normally
        startButton.setTitleColor(.white, for: .normal)
        startButton.titleLabel?.font = UIFont.systemFont(ofSize: 19, weight: .bold)
        startButton.contentHorizontalAlignment = .center

        // Translucent glass-style background
        startButton.backgroundColor = UIColor.white.withAlphaComponent(0.42)
        startButton.layer.cornerRadius = 32

        // Subtle glass border
        startButton.layer.borderColor = UIColor.white.withAlphaComponent(0.20).cgColor
        startButton.layer.borderWidth = 1

        // Soft floating shadow
        startButton.layer.shadowColor = UIColor.black.cgColor
        startButton.layer.shadowRadius = 10
        startButton.layer.shadowOpacity = 0.25
        startButton.layer.shadowOffset = CGSize(width: 0, height: 4)

        startButton.addTarget(self, action: #selector(playTapped), for: .touchUpInside)

        // Red recording dot
        let dotDiameter: CGFloat = 16
        let dot = UIView()
        dot.translatesAutoresizingMaskIntoConstraints = false
        dot.backgroundColor = .systemRed
        dot.layer.cornerRadius = dotDiameter / 2
        dot.isUserInteractionEnabled = false
        startButton.addSubview(dot)

        // Pulse animation
        let pulse = CABasicAnimation(keyPath: "transform.scale")
        pulse.fromValue = 1.0
        pulse.toValue = 1.3
        pulse.duration = 1.1
        pulse.autoreverses = true
        pulse.repeatCount = .infinity
        pulse.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        dot.layer.add(pulse, forKey: "pulse")

        NSLayoutConstraint.activate([
            dot.leadingAnchor.constraint(equalTo: startButton.leadingAnchor, constant: 20),
            dot.centerYAnchor.constraint(equalTo: startButton.centerYAnchor),
            dot.widthAnchor.constraint(equalToConstant: dotDiameter),
            dot.heightAnchor.constraint(equalToConstant: dotDiameter)
        ])
    }

    // MARK: - Recordings display

    @objc private func storeChanged() {
        DispatchQueue.main.async { self.reloadRecordings() }
    }

    private func reloadRecordings() {
        recordings = SessionStore.shared.sessions.sorted(by: { $0.createdAt > $1.createdAt })

        let hasRecordings = !recordings.isEmpty
        recordingsEmptyLabel.isHidden = hasRecordings
        lastSessionLabel.isHidden = !hasRecordings
        recordingsSeeAllButton.isHidden = !hasRecordings
        lastSessionThumbnail.isHidden = !hasRecordings
        statsStackView.isHidden = !hasRecordings

        guard let mostRecent = recordings.first else { return }

        _ = ordinalWithinDay(for: mostRecent)
        lastSessionLabel.text = "Last Session: \(formatSessionDate(mostRecent.createdAt, short: true))"

        if let url = mostRecent.thumbnailURL, let image = UIImage(contentsOfFile: url.path) {
            lastSessionThumbnail.image = image
        } else {
            lastSessionThumbnail.image = nil
        }

        // Footer stats reflect the most recent session (all-time top speed now lives in Settings).
        totalShotsColumn.setValue("\(mostRecent.kickCount)")
        topSpeedColumn.setValue(formatSpeed(mostRecent.topSpeed))
    }

    // Chronological position (1-based) of `record` among sessions from the same calendar day.
    private func ordinalWithinDay(for record: SessionRecord) -> Int {
        let cal = Calendar.current
        let dayStart = cal.startOfDay(for: record.createdAt)
        guard let dayEnd = cal.date(byAdding: .day, value: 1, to: dayStart) else { return 1 }
        let sameDay = SessionStore.shared.sessions
            .filter { $0.createdAt >= dayStart && $0.createdAt < dayEnd }
            .sorted { $0.createdAt < $1.createdAt }
        return (sameDay.firstIndex(where: { $0.id == record.id }) ?? 0) + 1
    }

    @objc private func lastSessionThumbnailTapped() {
        guard let record = recordings.first else { return }
        guard FileManager.default.fileExists(atPath: record.recordingFileURL.path) else {
            reloadRecordings()
            return
        }
        GameManager.shared.recordedVideoSource = AVAsset(url: record.recordingFileURL)
        // Mark this as a replay of an existing recording so Summary can offer to update its stats.
        GameManager.shared.replayingRecordID = record.id
        if let sp = storyboard?.instantiateViewController(withIdentifier: "SourcePickerViewController") {
            navigationController?.pushViewController(sp, animated: true)
        }
    }

    // MARK: - Actions

    private func animateHero() {
        let views: [UIView] = [logoImageView, titleLabel, subtitleLabel]
        for (index, v) in views.enumerated() {
            UIView.animate(withDuration: 0.6, delay: 0.1 * Double(index), options: [.curveEaseOut]) {
                v.alpha = 1
            }
        }
    }

    @objc private func howItWorksTapped() {
        let vc = HowItWorksViewController()
        navigationController?.pushViewController(vc, animated: true)
    }

    @objc private func playTapped() {
        // Three flows based on state:
        //   1) Dev mode ON  → Setup Instructions → SourcePicker (user chooses source)
        //   2) No recordings, non-dev → Setup Instructions → SourcePicker (auto-forwards to live)
        //   3) Has recordings, non-dev → SourcePicker directly (auto-forwards to live)
        // The single decision point in every flow is SourcePicker's viewDidAppear — it checks
        // directToLiveCamera and either shows its choice screen or auto-forwards.
        let devMode = SettingsStore.shared.developerMode
        let hasRecordings = !SessionStore.shared.sessions.isEmpty

        if devMode {
            GameManager.shared.directToLiveCamera = false
            pushStoryboardVC(withIdentifier: "MainViewController")
        } else if !hasRecordings {
            // First-time user still sees the setup instructions but skips the source-picker choice.
            GameManager.shared.directToLiveCamera = true
            pushStoryboardVC(withIdentifier: "MainViewController")
        } else {
            // Returning non-dev user: straight through — no setup instructions, no picker choice.
            GameManager.shared.directToLiveCamera = true
            pushStoryboardVC(withIdentifier: "SourcePickerViewController")
        }
    }

    private func pushStoryboardVC(withIdentifier identifier: String) {
        guard let vc = storyboard?.instantiateViewController(withIdentifier: identifier) else { return }
        navigationController?.pushViewController(vc, animated: true)
    }

    @objc private func recordingsTapped() {
        let vc = RecordingsViewController()
        navigationController?.pushViewController(vc, animated: true)
    }

    @objc private func settingsTapped() {
        let vc = SettingsViewController()
        navigationController?.pushViewController(vc, animated: true)
    }
}

// MARK: - HomeStatColumn
// Two-line stacked column used in the all-time stats footer: small monospace title on top,
// large white value below. Instances are laid out horizontally by `statsStackView`.

private class HomeStatColumn: UIStackView {
    private let titleLabel = UILabel()
    private let valueLabel = UILabel()

    init(title: String) {
        super.init(frame: .zero)
        translatesAutoresizingMaskIntoConstraints = false
        axis = .vertical
        alignment = .center
        spacing = 4

        titleLabel.text = title
        titleLabel.font = UIFont.monospacedSystemFont(ofSize: 11, weight: .medium)
        titleLabel.textColor = UIColor.white.withAlphaComponent(0.65)
        titleLabel.textAlignment = .center
        titleLabel.numberOfLines = 2
        titleLabel.adjustsFontSizeToFitWidth = true
        titleLabel.minimumScaleFactor = 0.85

        valueLabel.font = UIFont.systemFont(ofSize: 22, weight: .semibold)
        valueLabel.textColor = .white
        valueLabel.textAlignment = .center
        valueLabel.adjustsFontSizeToFitWidth = true
        valueLabel.minimumScaleFactor = 0.6

        addArrangedSubview(titleLabel)
        addArrangedSubview(valueLabel)
    }

    required init(coder: NSCoder) { fatalError("init(coder:) not implemented") }

    func setValue(_ text: String) { valueLabel.text = text }
}

// MARK: - GradientOverlayView
// Full-screen vertical gradient using a CAGradientLayer.

private class GradientOverlayView: UIView {
    var topColor: UIColor = .clear { didSet { updateColors() } }
    var bottomColor: UIColor = .clear { didSet { updateColors() } }

    override class var layerClass: AnyClass { CAGradientLayer.self }
    private var gradientLayer: CAGradientLayer { layer as! CAGradientLayer }

    override init(frame: CGRect) {
        super.init(frame: frame)
        gradientLayer.startPoint = CGPoint(x: 0.5, y: 0)
        gradientLayer.endPoint = CGPoint(x: 0.5, y: 1)
        isUserInteractionEnabled = false
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) not implemented") }

    private func updateColors() {
        gradientLayer.colors = [topColor.cgColor, bottomColor.cgColor]
    }
}

// MARK: - GradientLabel
// A UILabel that renders its text through a vertical gradient. Used for a subtle "shine" on the title.

private class GradientLabel: UILabel {
    var gradientColors: [CGColor] = [UIColor.white.cgColor] {
        didSet { setNeedsDisplay() }
    }

    override func drawText(in rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext() else {
            super.drawText(in: rect)
            return
        }
        UIGraphicsBeginImageContextWithOptions(bounds.size, false, 0)
        super.drawText(in: rect)
        let textImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        guard let textCGImage = textImage?.cgImage else {
            super.drawText(in: rect)
            return
        }

        context.saveGState()
        context.translateBy(x: 0, y: bounds.height)
        context.scaleBy(x: 1, y: -1)
        context.clip(to: bounds, mask: textCGImage)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        if let gradient = CGGradient(colorsSpace: colorSpace, colors: gradientColors as CFArray, locations: nil) {
            context.drawLinearGradient(
                gradient,
                start: CGPoint(x: bounds.midX, y: 0),
                end: CGPoint(x: bounds.midX, y: bounds.height),
                options: []
            )
        }
        context.restoreGState()
    }
}

// MARK: - How to Play screen
// Colocated with HomeViewController so it doesn't need its own file added to the Xcode project.
// Presents instructions for setting up + playing, with an inline [recordings] link that navigates
// to the Recordings screen. Uses the app's visual language: Inter for prose, SF Mono for
// bracketed "code-style" accents, bright green as the accent color.

class HowItWorksViewController: UIViewController, UITextViewDelegate {

    // App-wide accent color for code-style bracketed links.
    static let accentGreen = UIColor(red: 0.55, green: 1.0, blue: 0.25, alpha: 1.0)

    private let scrollView = UIScrollView()
    private let contentStack = UIStackView()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        title = "How to Play"
        setupBackground()
        setupContent()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        navigationController?.setNavigationBarHidden(false, animated: animated)
    }

    private func setupBackground() {
        let background = UIImageView(image: UIImage(named: "appbackground"))
        background.translatesAutoresizingMaskIntoConstraints = false
        background.contentMode = .scaleAspectFill
        background.clipsToBounds = true
        view.addSubview(background)

        // Heavier gradient overlay than the home screen — this is an instructional page and readability
        // matters more than showing off the background image.
        let overlay = HowToPlayGradientOverlay()
        overlay.translatesAutoresizingMaskIntoConstraints = false
        overlay.topColor = UIColor.black.withAlphaComponent(0.70)
        overlay.bottomColor = UIColor.black.withAlphaComponent(0.55)
        view.addSubview(overlay)

        NSLayoutConstraint.activate([
            background.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            background.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            background.topAnchor.constraint(equalTo: view.topAnchor),
            background.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            overlay.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            overlay.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            overlay.topAnchor.constraint(equalTo: view.topAnchor),
            overlay.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
    }

    private func setupContent() {
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.showsVerticalScrollIndicator = false
        view.addSubview(scrollView)

        contentStack.translatesAutoresizingMaskIntoConstraints = false
        contentStack.axis = .vertical
        contentStack.spacing = 20
        contentStack.alignment = .fill
        scrollView.addSubview(contentStack)

        NSLayoutConstraint.activate([
            scrollView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            scrollView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            scrollView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),

            contentStack.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 20),
            contentStack.bottomAnchor.constraint(equalTo: scrollView.bottomAnchor, constant: -20),
            contentStack.leadingAnchor.constraint(equalTo: scrollView.leadingAnchor, constant: 20),
            contentStack.trailingAnchor.constraint(equalTo: scrollView.trailingAnchor, constant: -20),
            contentStack.widthAnchor.constraint(equalTo: scrollView.widthAnchor, constant: -40)
        ])

        // 1. Intro paragraph
        contentStack.addArrangedSubview(makeBodyLabel(
            "To setup, place your phone adjacent to your shooting position and the goal."
        ))

        // 2. Field diagram
        let diagram = FieldDiagramView()
        contentStack.addArrangedSubview(diagram)

        // 3. Small italic note under the diagram
        contentStack.addArrangedSubview(makeSmallItalicLabel(
            "* For advanced players, this will be 20-25 yards away from the goal."
        ))

        // 4. Session end paragraph
        contentStack.addArrangedSubview(makeBodyLabel(
            "A session concludes once the screen is tapped twice or after 8 shots."
        ))

        // 5. Summary screen placeholder (freekickExample GIF for now)
        let summaryImageView = UIImageView()
        summaryImageView.translatesAutoresizingMaskIntoConstraints = false
        summaryImageView.contentMode = .scaleAspectFit
        summaryImageView.clipsToBounds = true
        summaryImageView.layer.cornerRadius = 10
        summaryImageView.loadGif(name: "freekickExample")
        NSLayoutConstraint.activate([
            summaryImageView.heightAnchor.constraint(equalTo: summaryImageView.widthAnchor, multiplier: 9.0 / 16.0)
        ])
        contentStack.addArrangedSubview(summaryImageView)

        // 6. Recordings paragraph — includes an inline [recordings] link.
        contentStack.addArrangedSubview(makeRecordingsParagraph())

        // 7. iOS screen recording note
        contentStack.addArrangedSubview(makeSmallItalicLabel(
            "* To save a video with Freekick's overlay, utilize iOS screen recording."
        ))

        // 8. Summary screen screenshot
        contentStack.addArrangedSubview(makeAspectFitImageView(named: "htp-summary"))

        // 9. Gameplay screenshot with all developer/extra-stats overlays visible.
        contentStack.addArrangedSubview(makeAspectFitImageView(named: "htp-settings-on"))

        // 10. Settings paragraph — includes an inline [settings] link.
        contentStack.addArrangedSubview(makeSettingsParagraph())
    }

    /// 16:9 image view with rounded corners, aspect-fit content mode. Height derives from the
    /// stack's chosen width via the ratio constraint so the images render at a consistent size
    /// regardless of the parent's actual width.
    private func makeAspectFitImageView(named name: String) -> UIImageView {
        let imageView = UIImageView(image: UIImage(named: name))
        imageView.translatesAutoresizingMaskIntoConstraints = false
        imageView.contentMode = .scaleAspectFit
        imageView.clipsToBounds = true
        imageView.layer.cornerRadius = 10
        NSLayoutConstraint.activate([
            imageView.heightAnchor.constraint(equalTo: imageView.widthAnchor, multiplier: 9.0 / 16.0)
        ])
        return imageView
    }

    // MARK: - Text component factories

    private func bodyFont(size: CGFloat = 15) -> UIFont {
        return UIFont(name: "Inter-Regular", size: size) ?? UIFont.systemFont(ofSize: size, weight: .regular)
    }

    private func monoFont(size: CGFloat = 15) -> UIFont {
        return UIFont.monospacedSystemFont(ofSize: size, weight: .semibold)
    }

    private func makeBodyLabel(_ text: String) -> UILabel {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.numberOfLines = 0
        label.text = text
        // Top-tier body text: 18pt bold, fully white. Sits ABOVE the italic subtext both
        // visually (page order) and in the type hierarchy.
        label.font = UIFont(name: "Inter-Bold", size: 18) ?? UIFont.systemFont(ofSize: 18, weight: .bold)
        label.textColor = .white
        return label
    }

    private func makeSmallItalicLabel(_ text: String) -> UILabel {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.numberOfLines = 0
        label.text = text
        // Subtext under the diagram: 14pt italic, white with a slight softening so it reads as
        // subordinate to the 18pt bold body above without disappearing.
        label.font = UIFont.italicSystemFont(ofSize: 14)
        label.textColor = UIColor.white.withAlphaComponent(0.85)
        return label
    }

    private func makeRecordingsParagraph() -> UITextView {
        return makeLinkedParagraph(
            prefix: "Raw footage is saved to Files → Freekick → Recordings, which can be played back in ",
            linkText: "[recordings]",
            linkURL: URL(string: "freekick://recordings")!
        )
    }

    private func makeSettingsParagraph() -> UITextView {
        return makeLinkedParagraph(
            prefix: "View extra stats and change unit system in ",
            linkText: "[settings]",
            linkURL: URL(string: "freekick://settings")!
        )
    }

    /// Body-styled paragraph with a monospace pill-link near the end. The link URL is opaque —
    /// it never actually navigates externally; the textView delegate intercepts taps and pushes
    /// the corresponding in-app screen. Ends with a period after the link.
    private func makeLinkedParagraph(prefix: String, linkText: String, linkURL: URL) -> UITextView {
        let textView = UITextView()
        textView.translatesAutoresizingMaskIntoConstraints = false
        textView.isEditable = false
        textView.isScrollEnabled = false
        textView.backgroundColor = .clear
        textView.textContainerInset = .zero
        textView.textContainer.lineFragmentPadding = 0
        textView.delegate = self
        textView.linkTextAttributes = [
            .foregroundColor: Self.accentGreen,
            .underlineStyle: 0
        ]

        let baseAttrs: [NSAttributedString.Key: Any] = [
            .font: bodyFont(),
            .foregroundColor: UIColor.white.withAlphaComponent(0.90)
        ]
        let linkAttrs: [NSAttributedString.Key: Any] = [
            .font: monoFont(),
            .foregroundColor: Self.accentGreen,
            .link: linkURL
        ]
        let attributed = NSMutableAttributedString(string: prefix, attributes: baseAttrs)
        attributed.append(NSAttributedString(string: linkText, attributes: linkAttrs))
        attributed.append(NSAttributedString(string: ".", attributes: baseAttrs))
        textView.attributedText = attributed
        return textView
    }

    // MARK: - UITextViewDelegate

    func textView(_ textView: UITextView, shouldInteractWith URL: URL, in characterRange: NSRange, interaction: UITextItemInteraction) -> Bool {
        switch URL.absoluteString {
        case "freekick://recordings":
            navigationController?.pushViewController(RecordingsViewController(), animated: true)
            return false
        case "freekick://settings":
            navigationController?.pushViewController(SettingsViewController(), animated: true)
            return false
        default:
            return true
        }
    }
}

// MARK: - FieldDiagramView
// The soccer pitch image with an overlaid phone marker + camera-FOV triangle drawn in code.
// The container view enforces the image's 5:7 aspect ratio so the overlay coordinates always
// line up with the underlying pitch, regardless of the view's rendered width.

private class FieldDiagramView: UIView {

    private let imageView = UIImageView()
    private let overlayLayer = CAShapeLayer()
    private let phoneLayer = CAShapeLayer()

    // Normalized (0…1) coords in the LANDSCAPE-oriented pitch. Phone at bottom-left corner,
    // vertical orientation. FOV cone opens up-and-to-the-right:
    //   - Upper edge grazes the left post of the goal at the top of the frame.
    //   - Lower edge extends past the bottom-right corner and exits off-screen (endpoint
    //     intentionally beyond bounds; clipped by view).
    private let phoneCenterNormalized = CGPoint(x: 0.08, y: 0.90)
    private let phoneSizeNormalized = CGSize(width: 0.03, height: 0.06)
    private let triangleLeg1Normalized = CGPoint(x: 0.58, y: 0.00)  // upper FOV edge
    private let triangleLeg2Normalized = CGPoint(x: 1.05, y: 0.76)  // lower FOV edge (past corner)

    init() {
        super.init(frame: .zero)
        translatesAutoresizingMaskIntoConstraints = false

        // Rotate the source portrait pitch 90° clockwise via the orientation flag — no pixel
        // copy needed, UIImageView respects the orientation and draws it landscape.
        if let src = UIImage(named: "soccerpitch"), let cg = src.cgImage {
            imageView.image = UIImage(cgImage: cg, scale: src.scale, orientation: .right)
        } else {
            imageView.image = UIImage(named: "soccerpitch")
        }
        imageView.translatesAutoresizingMaskIntoConstraints = false
        imageView.contentMode = .scaleToFill
        imageView.clipsToBounds = true
        imageView.layer.cornerRadius = 12
        addSubview(imageView)

        // Landscape aspect ratio: width > height. Source is 500×700 portrait; rotated it renders
        // as 700×500 landscape. Container constrained to 7:5 W:H (height = width × 5/7).
        NSLayoutConstraint.activate([
            imageView.leadingAnchor.constraint(equalTo: leadingAnchor),
            imageView.trailingAnchor.constraint(equalTo: trailingAnchor),
            imageView.topAnchor.constraint(equalTo: topAnchor),
            imageView.bottomAnchor.constraint(equalTo: bottomAnchor),
            heightAnchor.constraint(equalTo: widthAnchor, multiplier: 5.0 / 7.0)
        ])

        // Triangle (FOV cone) drawn under the phone marker so the marker sits on top.
        overlayLayer.fillColor = HowItWorksViewController.accentGreen.withAlphaComponent(0.25).cgColor
        overlayLayer.strokeColor = HowItWorksViewController.accentGreen.withAlphaComponent(0.9).cgColor
        overlayLayer.lineWidth = 1.5
        overlayLayer.lineJoin = .round
        layer.addSublayer(overlayLayer)

        // Phone marker on top.
        phoneLayer.fillColor = UIColor.white.cgColor
        phoneLayer.strokeColor = HowItWorksViewController.accentGreen.cgColor
        phoneLayer.lineWidth = 1.5
        layer.addSublayer(phoneLayer)

        layer.cornerRadius = 12
        clipsToBounds = true
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) not implemented") }

    override func layoutSubviews() {
        super.layoutSubviews()
        let w = bounds.width
        let h = bounds.height
        guard w > 0, h > 0 else { return }

        // Convert normalized coords to view-space.
        let phoneCenter = CGPoint(x: phoneCenterNormalized.x * w, y: phoneCenterNormalized.y * h)
        let phoneSize = CGSize(width: phoneSizeNormalized.width * w, height: phoneSizeNormalized.height * h)
        let leg1 = CGPoint(x: triangleLeg1Normalized.x * w, y: triangleLeg1Normalized.y * h)
        let leg2 = CGPoint(x: triangleLeg2Normalized.x * w, y: triangleLeg2Normalized.y * h)

        // FOV cone
        let tri = UIBezierPath()
        tri.move(to: phoneCenter)
        tri.addLine(to: leg1)
        tri.addLine(to: leg2)
        tri.close()
        overlayLayer.path = tri.cgPath

        // Phone marker — small vertical rounded rect centered on the phone position.
        let phoneRect = CGRect(
            x: phoneCenter.x - phoneSize.width / 2,
            y: phoneCenter.y - phoneSize.height / 2,
            width: phoneSize.width,
            height: phoneSize.height
        )
        phoneLayer.path = UIBezierPath(roundedRect: phoneRect, cornerRadius: phoneSize.width * 0.3).cgPath
    }
}

// MARK: - HowToPlayGradientOverlay
// A dedicated gradient overlay so we don't collide with the private GradientOverlayView above.

private class HowToPlayGradientOverlay: UIView {
    var topColor: UIColor = .clear { didSet { updateColors() } }
    var bottomColor: UIColor = .clear { didSet { updateColors() } }

    override class var layerClass: AnyClass { CAGradientLayer.self }
    private var gradientLayer: CAGradientLayer { layer as! CAGradientLayer }

    override init(frame: CGRect) {
        super.init(frame: frame)
        gradientLayer.startPoint = CGPoint(x: 0.5, y: 0)
        gradientLayer.endPoint = CGPoint(x: 0.5, y: 1)
        isUserInteractionEnabled = false
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) not implemented") }

    private func updateColors() {
        gradientLayer.colors = [topColor.cgColor, bottomColor.cgColor]
    }
}

// MARK: - Settings screen
// Colocated with HomeViewController so it doesn't need its own file added to the Xcode project.
// Two settings today: Developer Mode + Units. Same aesthetic as How to Play (dark gradient over
// the app background). Both toggles are backed by SettingsStore.

class SettingsViewController: UIViewController {

    private let developerToggle = UISwitch()
    private let unitsToggle = UISwitch()
    private let extraStatsToggle = UISwitch()
    // All-time top speed readout, sourced from every recorded session's `topSpeed`.
    private let topSpeedAllTimeColumn = HomeStatColumn(title: "Top Speed (All-time)")

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        title = "Settings"
        setupBackground()
        setupContent()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        navigationController?.setNavigationBarHidden(false, animated: animated)
        // Reflect current store values in case they were changed elsewhere.
        developerToggle.isOn = SettingsStore.shared.developerMode
        unitsToggle.isOn = SettingsStore.shared.useMetricUnits
        extraStatsToggle.isOn = SettingsStore.shared.showExtraStats
        refreshTopSpeedAllTime()
    }

    private func refreshTopSpeedAllTime() {
        let top = SessionStore.shared.sessions.map { $0.topSpeed }.max() ?? 0
        topSpeedAllTimeColumn.setValue(formatSpeed(top))
    }

    private func setupBackground() {
        let background = UIImageView(image: UIImage(named: "appbackground"))
        background.translatesAutoresizingMaskIntoConstraints = false
        background.contentMode = .scaleAspectFill
        background.clipsToBounds = true
        view.addSubview(background)

        let overlay = HowToPlayGradientOverlay()
        overlay.translatesAutoresizingMaskIntoConstraints = false
        overlay.topColor = UIColor.black.withAlphaComponent(0.70)
        overlay.bottomColor = UIColor.black.withAlphaComponent(0.55)
        view.addSubview(overlay)

        NSLayoutConstraint.activate([
            background.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            background.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            background.topAnchor.constraint(equalTo: view.topAnchor),
            background.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            overlay.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            overlay.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            overlay.topAnchor.constraint(equalTo: view.topAnchor),
            overlay.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
    }

    private func setupContent() {
        let stack = UIStackView()
        stack.translatesAutoresizingMaskIntoConstraints = false
        stack.axis = .vertical
        stack.spacing = 40
        stack.alignment = .fill
        view.addSubview(stack)

        stack.addArrangedSubview(makeSettingCard(
            title: "Units",
            description: "Switch from miles per hour (US) to kilometers per hour (UK). Applies to every speed reading across the app. This also changes the date format.",
            toggle: unitsToggle,
            initialState: SettingsStore.shared.useMetricUnits,
            action: #selector(unitsToggled(_:))
        ))

        stack.addArrangedSubview(makeSettingCard(
            title: "Extra Stats",
            description: "Show an in-game stats overlay with shot angle, time to goal, and distance.",
            toggle: extraStatsToggle,
            initialState: SettingsStore.shared.showExtraStats,
            action: #selector(extraStatsToggled(_:))
        ))

        stack.addArrangedSubview(makeSettingCard(
            title: "Developer Mode",
            description: "Show extra data during gameplay including kick classifier probabilities, pose tracking overlays, and detailed KPI readouts after each shot.",
            toggle: developerToggle,
            initialState: SettingsStore.shared.developerMode,
            action: #selector(developerModeToggled(_:))
        ))
        
        // Bottom-anchored all-time top speed readout — relocated here from the Home screen.
        view.addSubview(topSpeedAllTimeColumn)

        NSLayoutConstraint.activate([
            stack.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            stack.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 24),
            stack.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -24),

            topSpeedAllTimeColumn.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            topSpeedAllTimeColumn.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -32)
        ])
    }

    private func makeSettingCard(
        title: String,
        description: String,
        toggle: UISwitch,
        initialState: Bool,
        action: Selector
    ) -> UIView {
        let container = UIView()
        container.translatesAutoresizingMaskIntoConstraints = false

        let titleLabel = UILabel()
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        titleLabel.text = title
        titleLabel.font = UIFont(name: "Inter-Bold", size: 20) ?? UIFont.systemFont(ofSize: 20, weight: .bold)
        titleLabel.textColor = .white
        container.addSubview(titleLabel)

        toggle.translatesAutoresizingMaskIntoConstraints = false
        toggle.isOn = initialState
        toggle.onTintColor = HowItWorksViewController.accentGreen
        toggle.addTarget(self, action: action, for: .valueChanged)
        container.addSubview(toggle)

        let descLabel = UILabel()
        descLabel.translatesAutoresizingMaskIntoConstraints = false
        descLabel.text = description
        // Match the Home screen subtitle: 15pt system regular, 85% white.
        descLabel.font = UIFont.systemFont(ofSize: 15, weight: .regular)
        descLabel.textColor = UIColor.white.withAlphaComponent(0.85)
        descLabel.numberOfLines = 0
        container.addSubview(descLabel)

        NSLayoutConstraint.activate([
            titleLabel.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            titleLabel.topAnchor.constraint(equalTo: container.topAnchor),
            titleLabel.trailingAnchor.constraint(lessThanOrEqualTo: toggle.leadingAnchor, constant: -12),

            toggle.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            toggle.centerYAnchor.constraint(equalTo: titleLabel.centerYAnchor),

            descLabel.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            descLabel.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            descLabel.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 8),
            descLabel.bottomAnchor.constraint(equalTo: container.bottomAnchor)
        ])
        return container
    }

    @objc private func developerModeToggled(_ sender: UISwitch) {
        SettingsStore.shared.developerMode = sender.isOn
    }

    @objc private func unitsToggled(_ sender: UISwitch) {
        SettingsStore.shared.useMetricUnits = sender.isOn
        // Re-format the on-screen readout so the new unit shows without navigating away.
        refreshTopSpeedAllTime()
    }

    @objc private func extraStatsToggled(_ sender: UISwitch) {
        SettingsStore.shared.showExtraStats = sender.isOn
    }
}

