/*
See LICENSE folder for this sample's licensing information.

Abstract:
View controller to show the game summary.
*/

import UIKit
import AVFoundation

class SummaryViewController: UIViewController {

    @IBOutlet weak var speedValue: UILabel!
    @IBOutlet weak var angleValue: UILabel!
    @IBOutlet weak var scoreValue: UILabel!
    @IBOutlet weak var backgroundImage: UIImageView!

    private let gameManager = GameManager.shared
    private let keepButton = UIButton(type: .system)
    private let finalizingLabel = UILabel()

    override func viewDidLoad() {
        super.viewDidLoad()
        self.updateUI()
        setupSessionControls()
        // Three source flows resolve here:
        //   1. Replaying a stored SessionRecord (replayingRecordID != nil) → "Update Stats"
        //      overwrites the metric fields on the existing record in place.
        //   2. Uploaded ad-hoc video (recordedVideoSource != nil, no record ID) → hide button.
        //      Save-uploaded-video was attempted but the doc-picker's temp copy URL isn't
        //      guaranteed to survive iOS temp cleanup between pick time and save time — the copy
        //      to Recordings silently fails. Would need a persistent staging area to fix.
        //   3. Live camera capture → "Save to Recordings" moves the tmp .mov file into place
        //      and creates a SessionRecord. Waits for the recorder-finish notification first.
        if gameManager.replayingRecordID != nil {
            keepButton.setTitle("Update Stats", for: .normal)
            keepButton.isHidden = false
            keepButton.isEnabled = true
            finalizingLabel.isHidden = true
        } else if gameManager.recordedVideoSource != nil {
            keepButton.isHidden = true
            finalizingLabel.isHidden = true
        } else {
            finalizingLabel.isHidden = false
            keepButton.isEnabled = false
        }
        NotificationCenter.default.addObserver(self, selector: #selector(recorderDidFinish(_:)), name: SessionRecorder.didFinishRecordingNotification, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(recorderDidFail(_:)), name: SessionRecorder.didFailRecordingNotification, object: nil)
    }

    private func updateUI() {
        let stats = gameManager.playerStats
        backgroundImage.image = gameManager.previewImage
        displayTrajectories()
        // Speed label attributed string
        let speedValueFont = [NSAttributedString.Key.font: UIFont.systemFont(ofSize: 28.0, weight: .bold)]
        let speedValueText = NSMutableAttributedString(string: "\(round(stats.avgSpeed * 100) / 100)", attributes: speedValueFont)
        let speedUnitFont = [NSAttributedString.Key.font: UIFont.systemFont(ofSize: 22.0, weight: .bold)]
        speedValueText.append(NSAttributedString(string: " MPH", attributes: speedUnitFont))

        // set attributed text on a UILabel
        speedValue.attributedText = speedValueText
        angleValue.text = "\(round(stats.avgReleaseAngle * 100) / 100)°"
        let score = NSMutableAttributedString(string: "\(stats.totalScore)", attributes: [.foregroundColor: UIColor.white])
        score.append(NSAttributedString(string: "/\(GameConstants.maxKicks * Scoring.fifteen.rawValue)", attributes: [.foregroundColor: UIColor.white.withAlphaComponent(0.65)]))
        scoreValue.attributedText = score
    }

    private func setupSessionControls() {
        finalizingLabel.translatesAutoresizingMaskIntoConstraints = false
        finalizingLabel.text = "Finalizing recording..."
        finalizingLabel.textColor = .white
        finalizingLabel.textAlignment = .center
        finalizingLabel.isHidden = true

        keepButton.translatesAutoresizingMaskIntoConstraints = false
        keepButton.setTitle("Save to Recordings", for: .normal)
        keepButton.titleLabel?.font = UIFont.boldSystemFont(ofSize: 17)
        keepButton.backgroundColor = UIColor.systemBlue
        keepButton.setTitleColor(.white, for: .normal)
        keepButton.layer.cornerRadius = 8
        keepButton.addTarget(self, action: #selector(keepTapped), for: .touchUpInside)

        view.addSubview(finalizingLabel)
        view.addSubview(keepButton)

        NSLayoutConstraint.activate([
            keepButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            keepButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            keepButton.heightAnchor.constraint(equalToConstant: 44),
            keepButton.widthAnchor.constraint(equalToConstant: 160),

            finalizingLabel.topAnchor.constraint(equalTo: keepButton.bottomAnchor, constant: 4),
            finalizingLabel.centerXAnchor.constraint(equalTo: keepButton.centerXAnchor)
        ])
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    @objc private func recorderDidFinish(_ n: Notification) {
        DispatchQueue.main.async {
            self.finalizingLabel.isHidden = true
            self.keepButton.isEnabled = true
        }
    }

    @objc private func recorderDidFail(_ n: Notification) {
        DispatchQueue.main.async {
            self.finalizingLabel.isHidden = true
        }
    }

    // MARK: - Save

    @objc private func keepTapped() {
        // Replay-of-existing-recording path: overwrite the stat fields on the existing record
        // rather than creating a new one. File, thumbnail, id, and createdAt all stay intact.
        if let existingID = gameManager.replayingRecordID {
            updateExistingRecordStats(id: existingID)
            return
        }

        // Move the tmp recording to Documents/Recordings/, write a thumbnail, and add the
        // SessionRecord to the store. This is the live-camera capture path.
        guard let tmpURL = gameManager.currentSessionURL,
              FileManager.default.fileExists(atPath: tmpURL.path) else {
            gameManager.hasSavedToRecordings = true
            (parent as? RootViewController)?.exitToMenu()
            return
        }

        let date = Date()
        let destination = SessionStore.shared.makeRecordingFileURL(for: date)
        try? FileManager.default.removeItem(at: destination)  // overwrite if a same-minute file exists
        do {
            try FileManager.default.moveItem(at: tmpURL, to: destination)
        } catch {
            // If the move fails there's no recording to reference — bail out without a manifest entry.
            gameManager.hasSavedToRecordings = true
            (parent as? RootViewController)?.exitToMenu()
            return
        }

        let sessionID = UUID()
        let thumbnailName = saveThumbnail(for: sessionID)
        let stats = gameManager.playerStats
        let record = SessionRecord(
            id: sessionID,
            createdAt: date,
            recordingFileName: destination.lastPathComponent,
            thumbnailFileName: thumbnailName,
            topSpeed: stats.topSpeed,
            avgSpeed: stats.avgSpeed,
            kickCount: stats.kickCount,
            totalScore: stats.totalScore,
            avgReleaseAngle: stats.avgReleaseAngle,
            duration: nil,
            isFavorite: false,
            exportedToCameraRoll: false
        )
        try? SessionStore.shared.addSession(record)

        gameManager.hasSavedToRecordings = true
        (parent as? RootViewController)?.exitToMenu()
    }

    /// Replay flow: preserve id/createdAt/file/thumbnail and overwrite only the metric fields
    /// with values freshly computed from the re-analyzed video.
    private func updateExistingRecordStats(id: UUID) {
        guard var record = SessionStore.shared.sessions.first(where: { $0.id == id }) else {
            (parent as? RootViewController)?.exitToMenu()
            return
        }
        let stats = gameManager.playerStats
        record.topSpeed = stats.topSpeed
        record.avgSpeed = stats.avgSpeed
        record.kickCount = stats.kickCount
        record.totalScore = stats.totalScore
        record.avgReleaseAngle = stats.avgReleaseAngle
        try? SessionStore.shared.updateSession(record)
        (parent as? RootViewController)?.exitToMenu()
    }

    private func saveThumbnail(for sessionID: UUID) -> String? {
        guard let data = gameManager.previewImage.jpegData(compressionQuality: 0.7) else { return nil }
        let name = SessionStore.shared.makeThumbnailFileName(for: sessionID)
        let url = SessionStore.thumbnailsDirectory.appendingPathComponent(name)
        do {
            try data.write(to: url, options: .atomic)
            return name
        } catch {
            return nil
        }
    }

    private func displayTrajectories() {
        let stats = gameManager.playerStats
        // Fetch saved kick paths from playerStats and draw each kick on a TrajectoryView.
        let paths = stats.kickPaths
        let frame = view.bounds
        for path in paths {
            let trajectoryView = TrajectoryView(frame: frame)
            trajectoryView.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview(trajectoryView)
            NSLayoutConstraint.activate([
                trajectoryView.leftAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leftAnchor, constant: 0),
                trajectoryView.rightAnchor.constraint(equalTo: view.safeAreaLayoutGuide.rightAnchor, constant: 0),
                trajectoryView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 0),
                trajectoryView.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: 0)
            ])
            trajectoryView.addPath(path)
        }
    }
}
