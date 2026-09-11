//
//  MainViewController.swift
//  Freekick
//

import UIKit
import AVFoundation

class MainViewController: UIViewController {
    @IBOutlet weak var imageView: UIImageView!

    override func viewDidLoad() {
        super.viewDidLoad()
        imageView.loadGif(name: "freekickExample")
        if let recordingsButton = findButton(withTitle: "Recordings", in: view) {
            recordingsButton.addTarget(self, action: #selector(showRecordings), for: .touchUpInside)
        }
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        // Setup screen always wants the back arrow visible.
        navigationController?.setNavigationBarHidden(false, animated: animated)
    }

    @objc private func showRecordings() {
        let vc = RecordingsViewController()
        navigationController?.pushViewController(vc, animated: true)
    }

    private func findButton(withTitle title: String, in view: UIView) -> UIButton? {
        for sub in view.subviews {
            if let btn = sub as? UIButton, btn.title(for: .normal) == title {
                return btn
            }
            if let found = findButton(withTitle: title, in: sub) {
                return found
            }
        }
        return nil
    }
}
