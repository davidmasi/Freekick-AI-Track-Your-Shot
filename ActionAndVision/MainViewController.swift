//
//  MainViewController.swift
//  Freekick
//
//  Created by David on 12/30/24.
//  Copyright © 2024 Apple. All rights reserved.
//  FEAR IS THE MIND KILLER

import UIKit

class MainViewController: UIViewController {
    @IBOutlet weak var imageView: UIImageView!
    override func viewDidLoad() {
        super.viewDidLoad()
        imageView.loadGif(name: "freekickExample")
    }
}
