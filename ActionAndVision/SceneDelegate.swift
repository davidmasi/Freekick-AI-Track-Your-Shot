/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The app's scene delegate object.
*/

import UIKit
import AVFoundation

class SceneDelegate: UIResponder, UIWindowSceneDelegate {

    var window: UIWindow?

    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
        // Force dark mode app-wide regardless of the user's system setting. Screens like Setup
        // Instructions and SourcePicker use systemBackgroundColor / systemGray6Color in the storyboard,
        // which would render white in light mode. Locking to dark keeps every screen visually consistent
        // with the dark home / how-to-play / settings screens.
        guard let windowScene = scene as? UIWindowScene else { return }
        if let existingWindow = windowScene.windows.first {
            window = existingWindow
        }
        window?.overrideUserInterfaceStyle = .dark

        // Cold-launch via "Open with Freekick" (Files app / other sharesheets): stash the video
        // on GameManager before Home appears. Home's viewDidAppear picks it up and routes into
        // the analysis flow via SourcePicker.
        if let urlContext = connectionOptions.urlContexts.first {
            acceptIncomingVideo(url: urlContext.url)
        }
    }

    func scene(_ scene: UIScene, openURLContexts URLContexts: Set<UIOpenURLContext>) {
        // Warm-launch (app already running) via "Open with Freekick".
        guard let urlContext = URLContexts.first else { return }
        acceptIncomingVideo(url: urlContext.url)
        // Force the nav stack back to Home so its viewDidAppear picks up the new source and
        // routes into the analysis flow. Interrupts whatever the user was doing — acceptable
        // for an explicit "Open with Freekick" gesture.
        if let nav = window?.rootViewController as? UINavigationController {
            nav.popToRootViewController(animated: true)
        }
    }

    /// Copy the incoming file into our tmp directory so the AVAsset can read it without needing
    /// the source URL's security scope maintained. If the copy fails (transient Files.app URL,
    /// permission issue), fall back to using the URL directly and keep the security scope open
    /// for the AVAsset's lifetime. Actual navigation happens on HomeViewController.viewDidAppear.
    private func acceptIncomingVideo(url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        let ext = url.pathExtension.isEmpty ? "mov" : url.pathExtension
        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent("shared_\(UUID().uuidString).\(ext)")

        let gm = GameManager.shared
        gm.reset()

        do {
            try FileManager.default.copyItem(at: url, to: dest)
            // Copy succeeded, we own the tmp file — release the source scope.
            if scoped { url.stopAccessingSecurityScopedResource() }
            gm.recordedVideoSource = AVAsset(url: dest)
        } catch {
            // Copy failed — read straight from the source URL and leak the scope for the
            // duration of the analysis. iOS releases it when the app terminates.
            gm.recordedVideoSource = AVAsset(url: url)
        }
    }
}

