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
