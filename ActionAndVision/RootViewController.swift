/*
See LICENSE folder for this sample’s licensing information.

Abstract:
This is a custom container view controller that is responsible for two things:
    1. Hosting the CameraViewController that presents video frames captured by camera or being read from video file
    2. Presentation and dismissal of overlay view controllers based on current game state
*/

import UIKit

class RootViewController: UIViewController {
    
    @IBOutlet weak var closeButton: UIButton!
    @IBOutlet weak var imageView: UIImageView!
    private var cameraViewController: CameraViewController!
    private var overlayParentView: UIView!
    private var overlayViewController: UIViewController!
    private let gameManager = GameManager.shared
    
    override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
        return .landscape
    }

    override var preferredInterfaceOrientationForPresentation: UIInterfaceOrientation {
        return .landscapeRight
    }
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Change orientation lock to landscape only for this controller
        if let appDelegate = UIApplication.shared.delegate as? AppDelegate {
            appDelegate.orientationLock = .landscape
        }
        cameraViewController = CameraViewController()
        cameraViewController.view.frame = view.bounds
        cameraViewController.view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        addChild(cameraViewController)
        cameraViewController.beginAppearanceTransition(true, animated: true)
        view.addSubview(cameraViewController.view)
        cameraViewController.endAppearanceTransition()
        cameraViewController.didMove(toParent: self)
        overlayParentView = UIView(frame: view.bounds)
        overlayParentView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(overlayParentView)
        NSLayoutConstraint.activate([
            overlayParentView.leftAnchor.constraint(equalTo: view.leftAnchor, constant: 0),
            overlayParentView.rightAnchor.constraint(equalTo: view.rightAnchor, constant: 0),
            overlayParentView.topAnchor.constraint(equalTo: view.topAnchor, constant: 0),
            overlayParentView.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: 0)
        ])
        
        startObservingStateChanges()
        // Make sure close button stays in front of other views.
        view.bringSubviewToFront(closeButton)
        // Intercept close button taps to provide a confirmation sheet when
        // the Summary screen is visible. Remove any storyboard targets
        // and add our own handler so we can decide behavior at runtime.
        closeButton.removeTarget(nil, action: nil, for: .touchUpInside)
        closeButton.addTarget(self, action: #selector(closeButtonTapped(_:)), for: .touchUpInside)
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        // Gameplay uses the on-screen X button for exit — nav bar back button would conflict.
        navigationController?.setNavigationBarHidden(true, animated: animated)
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        gameManager.stateMachine.enter(GameManager.SetupCameraState.self)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        // Restore the app's default orientation lock (portrait) when leaving gameplay. The rest
        // of the app is portrait-only per AppDelegate.orientationLock.
        if let appDelegate = UIApplication.shared.delegate as? AppDelegate {
            appDelegate.orientationLock = .portrait
        }
    }
    
    private func presentOverlayViewController(_ newOverlayViewController: UIViewController?, completion: (() -> Void)?) {
        defer {
            completion?()
        }
        
        guard overlayViewController != newOverlayViewController else {
            return
        }
        
        if let currentOverlay = overlayViewController {
            currentOverlay.willMove(toParent: nil)
            currentOverlay.beginAppearanceTransition(false, animated: true)
            currentOverlay.view.removeFromSuperview()
            currentOverlay.endAppearanceTransition()
            currentOverlay.removeFromParent()
        }
        
        if let newOverlay = newOverlayViewController {
            newOverlay.view.frame = overlayParentView.bounds
            newOverlay.view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
            addChild(newOverlay)
            newOverlay.beginAppearanceTransition(true, animated: true)
            overlayParentView.addSubview(newOverlay.view)
            newOverlay.endAppearanceTransition()
            newOverlay.didMove(toParent: self)
        }
        
        overlayViewController = newOverlayViewController
    }

}

// MARK: - Handle states that require view controller transitions

// This is where the overlay controllers management happens.
extension RootViewController: GameStateChangeObserver {
    func gameManagerDidEnter(state: GameManager.State, from previousState: GameManager.State?) {
        // Create an overlay view controller based on the game state
        let controllerToPresent: UIViewController
        switch state {
        case is GameManager.DetectingGoalState:
            controllerToPresent = SetupViewController()
        case is GameManager.DetectingPlayerState:
            controllerToPresent = GameViewController()
        case is GameManager.ShowSummaryState:
            controllerToPresent = SummaryViewController()
        default:
            //The new state does not require new view controller, so just return.
            return
        }
        
        // Remove existing overlay controller (if any) from game manager listeners
        if let currentListener = overlayViewController as? GameStateChangeObserverViewController {
            currentListener.stopObservingStateChanges()
        }
        
        presentOverlayViewController(controllerToPresent) {
            //Adjust safe area insets on overlay controller to match actual video outpput area.
            if let cameraVC = self.cameraViewController {
                let viewRect = cameraVC.view.frame
                let videoRect = cameraVC.viewRectForVisionRect(CGRect(x: 0, y: 0, width: 1, height: 1))
                let insets = controllerToPresent.view.safeAreaInsets
                let additionalInsets = UIEdgeInsets(
                        top: videoRect.minY - viewRect.minY - insets.top,
                        left: videoRect.minX - viewRect.minX - insets.left,
                        bottom: viewRect.maxY - videoRect.maxY - insets.bottom,
                        right: viewRect.maxX - videoRect.maxX - insets.right)
                controllerToPresent.additionalSafeAreaInsets = additionalInsets
            }

            // If new overlay controller conforms to GameManagerListener, add it to the listeners.
            if let gameManagerListener = controllerToPresent as? GameStateChangeObserverViewController {
                gameManagerListener.startObservingStateChanges()
            }
            
            // If new overlay controller conforms to CameraViewControllerOutputDelegate
            // set it as a CameraViewController's delegate, so it can process the frames
            // that are coming from the live camera preview or being read from pre-recorded video file.
            if let outputDelegate = controllerToPresent as? CameraViewControllerOutputDelegate {
                self.cameraViewController.outputDelegate = outputDelegate
            }
        }
    }
}

// MARK: - Close button handling
extension RootViewController {
    func exitToMenu() {
        // If a live-camera recording was made but never committed to Recordings, delete the tmp file.
        // Without this, .mov files accumulate in tmp/ until iOS cleans them up.
        if !gameManager.hasSavedToRecordings,
           let url = gameManager.currentSessionURL,
           FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.removeItem(at: url)
        }
        gameManager.reset()
        navigationController?.popViewController(animated: true)
    }

    @objc private func closeButtonTapped(_ sender: UIButton) {
        let onSummary = overlayViewController is SummaryViewController
        let shouldWarn = onSummary
            && !gameManager.hasSavedToRecordings
            && gameManager.recordedVideoSource == nil

        if shouldWarn {
            presentUnsavedWarning()
        } else {
            exitToMenu()
        }
    }

    private func presentUnsavedWarning() {
        let alert = UIAlertController(
            title: "Unsaved Recording",
            message: "You haven't saved this recording.",
            preferredStyle: .alert)

        alert.addAction(UIAlertAction(title: "Save to Recordings", style: .default, handler: { _ in
            self.gameManager.hasSavedToRecordings = true
            self.exitToMenu()
        }))

        alert.addAction(UIAlertAction(title: "Discard", style: .destructive, handler: { _ in
            self.exitToMenu()
        }))

        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))

        present(alert, animated: true, completion: nil)
    }

}
