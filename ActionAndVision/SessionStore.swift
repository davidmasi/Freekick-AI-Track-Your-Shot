import Foundation

struct SessionRecord: Codable, Identifiable, Equatable {
    let id: UUID
    let createdAt: Date
    let recordingFileName: String
    let thumbnailFileName: String?
    var topSpeed: Double
    var avgSpeed: Double
    var kickCount: Int
    var totalScore: Int
    var avgReleaseAngle: Double
    var duration: TimeInterval?
    var isFavorite: Bool
    var exportedToCameraRoll: Bool

    var recordingFileURL: URL {
        return SessionStore.recordingsDirectory.appendingPathComponent(recordingFileName)
    }

    var thumbnailURL: URL? {
        guard let thumbnailFileName = thumbnailFileName else { return nil }
        return SessionStore.thumbnailsDirectory.appendingPathComponent(thumbnailFileName)
    }
}

extension Notification.Name {
    static let sessionStoreDidChange = Notification.Name("SessionStoreDidChange")
}

enum SessionStoreError: Error {
    case failedToCreateDirectory(URL)
    case failedToWriteManifest(Error)
    case failedToLoadManifest(Error)
}

final class SessionStore {
    static let shared = SessionStore()

    private let ioQueue = DispatchQueue(label: "com.freekick.sessionstore")

    // User-visible: saved recordings end up here, exposed via Files app.
    static let recordingsDirectory: URL = {
        return documentsDirectory.appendingPathComponent("Recordings", isDirectory: true)
    }()

    // System-managed scratch space for in-progress recordings. iOS may purge under pressure.
    static let tempRecordingsDirectory: URL = {
        return FileManager.default.temporaryDirectory
    }()

    // Internal app data — invisible to user, persisted between launches.
    static let thumbnailsDirectory: URL = {
        return applicationSupportDirectory.appendingPathComponent("Thumbnails", isDirectory: true)
    }()

    private static let manifestFileName = "sessions.json"
    private static let documentsDirectory: URL = {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    }()
    private static let applicationSupportDirectory: URL = {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
    }()

    private(set) var sessions: [SessionRecord] = []

    private var manifestURL: URL {
        return Self.applicationSupportDirectory.appendingPathComponent(Self.manifestFileName)
    }

    private init() {
        ioQueue.sync {
            do {
                try createDirectoriesIfNeeded()
                try loadManifest()
            } catch {
                sessions = []
            }
        }
    }

    func addSession(_ session: SessionRecord) throws {
        var thrown: Error?
        ioQueue.sync {
            sessions.append(session)
            do {
                try saveManifest()
                DispatchQueue.main.async {
                    NotificationCenter.default.post(name: .sessionStoreDidChange, object: self)
                }
            } catch {
                thrown = error
            }
        }
        if let e = thrown { throw e }
    }

    func updateSession(_ session: SessionRecord) throws {
        var thrown: Error?
        ioQueue.sync {
            guard let index = sessions.firstIndex(where: { $0.id == session.id }) else { return }
            sessions[index] = session
            do {
                try saveManifest()
                DispatchQueue.main.async {
                    NotificationCenter.default.post(name: .sessionStoreDidChange, object: self)
                }
            } catch {
                thrown = error
            }
        }
        if let e = thrown { throw e }
    }

    func removeSession(_ session: SessionRecord) throws {
        var thrown: Error?
        ioQueue.sync {
            sessions.removeAll { $0.id == session.id }
            do {
                try saveManifest()
            } catch {
                thrown = error
            }
        }
        // Each SessionRecord maps 1:1 to a recording + thumbnail — remove both on delete.
        try? FileManager.default.removeItem(at: session.recordingFileURL)
        if let thumbnailURL = session.thumbnailURL {
            try? FileManager.default.removeItem(at: thumbnailURL)
        }
        DispatchQueue.main.async {
            NotificationCenter.default.post(name: .sessionStoreDidChange, object: self)
        }
        if let e = thrown { throw e }
    }

    private func createDirectoriesIfNeeded() throws {
        let fileManager = FileManager.default
        let directories = [
            Self.applicationSupportDirectory,
            Self.recordingsDirectory,
            Self.thumbnailsDirectory
        ]
        for directory in directories {
            guard !fileManager.fileExists(atPath: directory.path) else { continue }
            do {
                try fileManager.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
            } catch {
                throw SessionStoreError.failedToCreateDirectory(directory)
            }
        }
    }

    private func loadManifest() throws {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: manifestURL.path) else {
            sessions = []
            return
        }
        do {
            let data = try Data(contentsOf: manifestURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let loaded = try decoder.decode([SessionRecord].self, from: data)
            // Drop entries whose .mov file has been deleted (e.g., user removed it via Files app).
            // Otherwise the Recordings screen would show phantom rows.
            sessions = loaded.filter { fileManager.fileExists(atPath: $0.recordingFileURL.path) }
            if sessions.count != loaded.count {
                try? saveManifest()
            }
        } catch {
            throw SessionStoreError.failedToLoadManifest(error)
        }
    }

    private func saveManifest() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        do {
            let data = try encoder.encode(sessions)
            try data.write(to: manifestURL, options: .atomic)
        } catch {
            throw SessionStoreError.failedToWriteManifest(error)
        }
    }

    // Recording is written here while in progress. iOS auto-cleans tmp/.
    func makeTempSessionFileURL() -> URL {
        let fileName = "session_\(UUID().uuidString).mov"
        return Self.tempRecordingsDirectory.appendingPathComponent(fileName)
    }

    // User-facing destination filename. We move tmp → here on Save to Recordings.
    func makeRecordingFileURL(for date: Date) -> URL {
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd_HHmm"
        let fileName = "freekick_\(df.string(from: date)).mov"
        return Self.recordingsDirectory.appendingPathComponent(fileName)
    }

    func makeThumbnailFileName(for sessionID: UUID) -> String {
        return "session_\(sessionID.uuidString)_thumb.jpg"
    }
}
