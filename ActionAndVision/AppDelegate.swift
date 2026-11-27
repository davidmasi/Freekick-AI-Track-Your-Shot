/*
See LICENSE folder for this sample’s licensing information.
#KnowYourShot
Abstract:
The app's delegate object.
*/

import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var orientationLock: UIInterfaceOrientationMask = .portrait

    func application(_ application: UIApplication,
                     supportedInterfaceOrientationsFor window: UIWindow?) -> UIInterfaceOrientationMask {
        return orientationLock  // Return the current orientation lock
    }

    func applicationDidFinishLaunching(_ application: UIApplication) {
        // Warmup Vision pipeline. This needs to be done just once.
        // Off the main thread: on a cold install, CoreML model compilation for the device's
        // Neural Engine can be slow enough to trip iOS's launch watchdog (~20s to return from
        // this method), which the OS reports as a crash on launch.
        DispatchQueue.global(qos: .userInitiated).async {
            warmUpVisionPipeline()
        }
    }

    func application(_ application: UIApplication,
                     configurationForConnecting connectingSceneSession: UISceneSession,
                     options: UIScene.ConnectionOptions) -> UISceneConfiguration {
        // Called when a new scene session is being created.
        // Use this method to select a configuration to create the new scene with.
        return UISceneConfiguration(name: "Default Configuration", sessionRole: connectingSceneSession.role)
    }
}
