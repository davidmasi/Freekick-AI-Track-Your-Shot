//
//  RecordingsViewController.swift
//  Freekick
//
//  Displays the user's saved live-camera sessions. Rows are sorted newest-first, each shows
//  date + inline stats + thumbnail with a "View" affordance. Tapping the thumbnail replays the
//  video through the analysis flow. Swipe-to-delete removes both the manifest entry and the .mov.
//

import UIKit
import AVFoundation

class RecordingsViewController: UIViewController {

    private var recordings: [SessionRecord] = []
    private let tableView = UITableView()
    private let emptyStateLabel = UILabel()
    // "Swipe to delete" hint pinned to the bottom safe area so it stays visible while the list
    // scrolls, rather than sitting below the last row via tableFooterView.
    private let swipeHintLabel = UILabel()

    // Date format is now derived per-row via formatSessionDate() so it flips between US
    // (MM-dd-yyyy) and UK (dd-MM-yyyy) with the Units toggle. No stored formatter.

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Recordings"
        view.backgroundColor = .black

        tableView.translatesAutoresizingMaskIntoConstraints = false
        tableView.backgroundColor = .black
        tableView.separatorColor = UIColor.white.withAlphaComponent(0.15)
        tableView.rowHeight = UITableView.automaticDimension
        tableView.estimatedRowHeight = 100
        tableView.register(RecordingCell.self, forCellReuseIdentifier: "cell")
        tableView.dataSource = self
        tableView.delegate = self
        view.addSubview(tableView)

        swipeHintLabel.translatesAutoresizingMaskIntoConstraints = false
        swipeHintLabel.text = "Swipe left on recording to delete"
        swipeHintLabel.textColor = UIColor.white.withAlphaComponent(0.4)
        swipeHintLabel.font = UIFont.systemFont(ofSize: 12)
        swipeHintLabel.textAlignment = .center
        swipeHintLabel.backgroundColor = .black
        swipeHintLabel.isHidden = true
        view.addSubview(swipeHintLabel)

        emptyStateLabel.translatesAutoresizingMaskIntoConstraints = false
        emptyStateLabel.text = "No recordings yet. Use the live camera to get started."
        emptyStateLabel.textColor = .white
        emptyStateLabel.textAlignment = .center
        emptyStateLabel.numberOfLines = 0
        emptyStateLabel.font = UIFont.systemFont(ofSize: 15)
        view.addSubview(emptyStateLabel)

        NSLayoutConstraint.activate([
            // Table fills from the top safe area down to the top of the hint. Doesn't extend
            // under the hint, so the last row is fully visible when scrolled to the bottom.
            tableView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            tableView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            tableView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            tableView.bottomAnchor.constraint(equalTo: swipeHintLabel.topAnchor),

            swipeHintLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            swipeHintLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            swipeHintLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
            swipeHintLabel.heightAnchor.constraint(equalToConstant: 32),

            emptyStateLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            emptyStateLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            emptyStateLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 40),
            emptyStateLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -40)
        ])

        NotificationCenter.default.addObserver(self, selector: #selector(storeChanged), name: .sessionStoreDidChange, object: nil)
        // Reload rows when the units toggle flips so avg/top speed cells re-render in the new unit.
        NotificationCenter.default.addObserver(self, selector: #selector(unitsChanged), name: SettingsStore.unitsDidChange, object: nil)
        loadRecordings()
    }

    @objc private func unitsChanged() {
        DispatchQueue.main.async { self.tableView.reloadData() }
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        navigationController?.setNavigationBarHidden(false, animated: animated)
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    @objc private func storeChanged() {
        DispatchQueue.main.async { self.loadRecordings() }
    }

    private func loadRecordings() {
        recordings = SessionStore.shared.sessions.sorted(by: { $0.createdAt > $1.createdAt })
        emptyStateLabel.isHidden = !recordings.isEmpty
        swipeHintLabel.isHidden = recordings.isEmpty
        tableView.reloadData()
    }

    private func confirmDelete(_ record: SessionRecord, completion: @escaping (Bool) -> Void) {
        let alert = UIAlertController(
            title: "Delete Recording?",
            message: "This removes the video file permanently.",
            preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: { _ in completion(false) }))
        alert.addAction(UIAlertAction(title: "Delete", style: .destructive, handler: { _ in
            try? SessionStore.shared.removeSession(record)
            completion(true)
        }))
        present(alert, animated: true)
    }

    // MARK: - Actions

    private func didPickRecording(_ record: SessionRecord) {
        // Guard against a file that was deleted externally between load and tap.
        guard FileManager.default.fileExists(atPath: record.recordingFileURL.path) else {
            loadRecordings()
            return
        }
        GameManager.shared.recordedVideoSource = AVAsset(url: record.recordingFileURL)
        // Mark this as a replay of an existing recording so Summary can offer to update its stats.
        GameManager.shared.replayingRecordID = record.id
        // Push SourcePicker directly — it will auto-forward to RootViewController when it sees
        // recordedVideoSource is already set. Instantiate the storyboard explicitly by name
        // because this VC is created programmatically — self.storyboard is nil here.
        let mainStoryboard = UIStoryboard(name: "Main", bundle: nil)
        let sp = mainStoryboard.instantiateViewController(withIdentifier: "SourcePickerViewController")
        navigationController?.pushViewController(sp, animated: true)
    }

}

extension RecordingsViewController: UITableViewDataSource, UITableViewDelegate {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return recordings.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "cell", for: indexPath) as! RecordingCell
        let record = recordings[indexPath.row]
        cell.configure(with: record)
        cell.onPlayTapped = { [weak self] in self?.didPickRecording(record) }
        return cell
    }

    func tableView(_ tableView: UITableView, trailingSwipeActionsConfigurationForRowAt indexPath: IndexPath) -> UISwipeActionsConfiguration? {
        let record = recordings[indexPath.row]
        let deleteAction = UIContextualAction(style: .destructive, title: "Delete") { [weak self] _, _, completion in
            self?.confirmDelete(record, completion: completion)
        }
        return UISwipeActionsConfiguration(actions: [deleteAction])
    }
}

// MARK: - Cell

private class RecordingCell: UITableViewCell {

    private let dateLabel = UILabel()
    // 2×2 stats grid: avg speed + top speed on the first row, total shots + score on the second.
    private let avgSpeedLabel = UILabel()
    private let topSpeedLabel = UILabel()
    private let totalShotsLabel = UILabel()
    private let scoreLabel = UILabel()
    private let thumbnailImageView = UIImageView()

    var onPlayTapped: (() -> Void)?

    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        setupViews()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupViews()
    }

    private func setupViews() {
        backgroundColor = .black
        selectionStyle = .none
        contentView.backgroundColor = .black

        dateLabel.translatesAutoresizingMaskIntoConstraints = false
        dateLabel.font = UIFont.boldSystemFont(ofSize: 16)
        dateLabel.textColor = .white
        contentView.addSubview(dateLabel)

        for label in [avgSpeedLabel, topSpeedLabel, totalShotsLabel, scoreLabel] {
            label.translatesAutoresizingMaskIntoConstraints = false
            label.font = UIFont.systemFont(ofSize: 13)
            label.textColor = UIColor.white.withAlphaComponent(0.85)
            contentView.addSubview(label)
        }

        thumbnailImageView.translatesAutoresizingMaskIntoConstraints = false
        thumbnailImageView.contentMode = .scaleAspectFill
        thumbnailImageView.clipsToBounds = true
        thumbnailImageView.layer.cornerRadius = 6
        thumbnailImageView.backgroundColor = UIColor.white.withAlphaComponent(0.1)
        thumbnailImageView.isUserInteractionEnabled = true
        // Tapping the thumbnail plays the recording.
        let playTap = UITapGestureRecognizer(target: self, action: #selector(handlePlayTap))
        thumbnailImageView.addGestureRecognizer(playTap)
        contentView.addSubview(thumbnailImageView)

        NSLayoutConstraint.activate([
            // Left column — date at top, 2×2 stats grid below.
            dateLabel.leadingAnchor.constraint(equalTo: contentView.layoutMarginsGuide.leadingAnchor),
            dateLabel.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 12),
            dateLabel.trailingAnchor.constraint(lessThanOrEqualTo: thumbnailImageView.leadingAnchor, constant: -12),

            // Row 1: Top Speed (left) | Total Shots (right of a fixed 175pt-offset column).
            topSpeedLabel.leadingAnchor.constraint(equalTo: dateLabel.leadingAnchor),
            topSpeedLabel.topAnchor.constraint(equalTo: dateLabel.bottomAnchor, constant: 8),

            totalShotsLabel.leadingAnchor.constraint(equalTo: topSpeedLabel.leadingAnchor, constant: 175),
            totalShotsLabel.centerYAnchor.constraint(equalTo: topSpeedLabel.centerYAnchor),
            totalShotsLabel.trailingAnchor.constraint(lessThanOrEqualTo: thumbnailImageView.leadingAnchor, constant: -12),

            // Row 2: Average Speed (left) | Score (right, same left offset as Total Shots).
            avgSpeedLabel.leadingAnchor.constraint(equalTo: dateLabel.leadingAnchor),
            avgSpeedLabel.topAnchor.constraint(equalTo: topSpeedLabel.bottomAnchor, constant: 4),
            avgSpeedLabel.bottomAnchor.constraint(lessThanOrEqualTo: contentView.bottomAnchor, constant: -12),

            scoreLabel.leadingAnchor.constraint(equalTo: totalShotsLabel.leadingAnchor),
            scoreLabel.centerYAnchor.constraint(equalTo: avgSpeedLabel.centerYAnchor),
            scoreLabel.trailingAnchor.constraint(lessThanOrEqualTo: thumbnailImageView.leadingAnchor, constant: -12),

            // Right column — thumbnail alone (no more View button above it).
            thumbnailImageView.trailingAnchor.constraint(equalTo: contentView.layoutMarginsGuide.trailingAnchor),
            thumbnailImageView.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
            thumbnailImageView.widthAnchor.constraint(equalToConstant: 110),
            thumbnailImageView.heightAnchor.constraint(equalTo: thumbnailImageView.widthAnchor, multiplier: 9.0 / 16.0),
            thumbnailImageView.topAnchor.constraint(greaterThanOrEqualTo: contentView.topAnchor, constant: 12),
            thumbnailImageView.bottomAnchor.constraint(lessThanOrEqualTo: contentView.bottomAnchor, constant: -12)
        ])
    }

    func configure(with record: SessionRecord) {
        dateLabel.text = formatSessionDate(record.createdAt, short: false)
        avgSpeedLabel.text = "Average Speed: \(formatSpeed(record.avgSpeed))"
        topSpeedLabel.text = "Top Speed: \(formatSpeed(record.topSpeed))"
        totalShotsLabel.text = "Shots: \(record.kickCount)"
        scoreLabel.text = "Score: \(record.totalScore)"
        if let url = record.thumbnailURL, let image = UIImage(contentsOfFile: url.path) {
            thumbnailImageView.image = image
        } else {
            thumbnailImageView.image = nil
        }
    }

    @objc private func handlePlayTap() { onPlayTapped?() }
}
