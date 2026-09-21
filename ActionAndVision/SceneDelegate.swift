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

        // Cold-launch via "Open with Freekick" (Files app / other sharesheets). Deferring the
        // navigation to the next main-runloop pass lets UIKit finish setting up the storyboard's
        // initial nav controller + Home first; pushing before that is a no-op.
        if let urlContext = connectionOptions.urlContexts.first {
            acceptIncomingVideo(url: urlContext.url)
            DispatchQueue.main.async { [weak self] in
                self?.routeToAnalysis(animated: false)
            }
        }
    }

    func scene(_ scene: UIScene, openURLContexts URLContexts: Set<UIOpenURLContext>) {
        // Warm-launch: the app is already running, the view hierarchy is up. Route immediately.
        guard let urlContext = URLContexts.first else { return }
        acceptIncomingVideo(url: urlContext.url)
        routeToAnalysis(animated: true)
    }

    /// Push SourcePicker onto the nav stack so its auto-forward logic picks up the just-set
    /// recordedVideoSource and segues into the analysis flow. Pops back to root first if the
    /// user was mid-navigation (say, in gameplay) — this is an explicit "Open with Freekick"
    /// gesture, so interrupting whatever was on screen is expected. Idempotent enough that
    /// firing it twice on the same launch would just push SourcePicker twice, but neither
    /// callsite hits it twice.
    private func routeToAnalysis(animated: Bool) {
        guard let nav = window?.rootViewController as? UINavigationController else { return }
        if nav.viewControllers.count > 1 {
            nav.popToRootViewController(animated: false)
        }
        let mainStoryboard = UIStoryboard(name: "Main", bundle: nil)
        if let sp = mainStoryboard.instantiateViewController(withIdentifier: "SourcePickerViewController") as? SourcePickerViewController {
            nav.pushViewController(sp, animated: animated)
        }
    }

    /// Copy the incoming file into our tmp directory so the AVAsset can read it without needing
    /// the source URL's security scope maintained. If the copy fails (transient Files.app URL,
    /// permission issue), fall back to using the URL directly and keep the security scope open
    /// for the AVAsset's lifetime.
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

