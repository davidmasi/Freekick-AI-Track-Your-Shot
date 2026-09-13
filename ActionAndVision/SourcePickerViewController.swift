/*
See LICENSE folder for this sample’s licensing information.

Abstract:
This view controller allows to choose the video source used by the app.
     It can be either a camera or a prerecorded video file.
*/

import UIKit
import AVFoundation
import PhotosUI

class SourcePickerViewController: UIViewController {

    private let gameManager = GameManager.shared

    override func viewDidLoad() {
        super.viewDidLoad()
        gameManager.stateMachine.enter(GameManager.InactiveState.self)
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        navigationController?.setNavigationBarHidden(false, animated: animated)
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        // Auto-forward when EITHER:
        //   - recordedVideoSource is set (Recordings flow, or upload)
        //   - directToLiveCamera is set (non-dev Start Session flow — see HomeViewController.playTapped)
        // Consume directToLiveCamera on the way through so it can't linger and mis-fire next time.
        let shouldForwardToLive = gameManager.directToLiveCamera
        if gameManager.recordedVideoSource != nil || shouldForwardToLive {
            if shouldForwardToLive { gameManager.directToLiveCamera = false }
            performSegue(withIdentifier: "ShowRootControllerSegue", sender: self)
            return
        }
        // Manual state (initial dev-mode landing, or a gameplay-cancel return trip):
        //   - Non-dev: auto-pop to Home. This is the second half of the two-step return from
        //     gameplay — RootVC pops to here (landscape → landscape, no rotation), and this
        //     pop takes us the rest of the way to Home where portrait rotation happens on a
        //     simpler screen than the live-camera view.
        //   - Dev mode: stay put so the developer can pick a different source. Strip Setup
        //     Instructions (MainViewController) from the back stack so long-pressing back
        //     from here only shows Home instead of both Home and Setup Instructions.
        if !SettingsStore.shared.developerMode {
            navigationController?.popToRootViewController(animated: true)
        } else if let nav = navigationController {
            let stack = nav.viewControllers
            let filtered = stack.filter { !($0 is MainViewController) }
            if filtered.count != stack.count {
                nav.setViewControllers(filtered, animated: false)
            }
        }
    }
    
    @IBAction func handleUploadVideoButton(_ sender: Any) {
        let docPicker = UIDocumentPickerViewController(forOpeningContentTypes: [.movie], asCopy: true)
        docPicker.delegate = self
        present(docPicker, animated: true)
    }
    
    @IBAction func revertToSourcePicker(_ segue: UIStoryboardSegue) {
        // This is for unwinding to this controller in storyboard.
        gameManager.reset()
    }
}

extension SourcePickerViewController: UIDocumentPickerDelegate {
    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
        gameManager.recordedVideoSource = nil
    }
    
    func  documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        guard let url = urls.first else {
            return
        }
        gameManager.recordedVideoSource = AVAsset(url: url)
        performSegue(withIdentifier: "ShowRootControllerSegue", sender: self)
    }
    
    override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
        return .landscape // Choose the desired orientation(s)
    }

    // Specify the preferred orientation when the view controller is presented
    override var preferredInterfaceOrientationForPresentation: UIInterfaceOrientation {
        return .landscapeRight // Choose the specific orientation
    }
}
