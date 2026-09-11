/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The app's scene delegate object.
*/

import UIKit

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
    }

}

