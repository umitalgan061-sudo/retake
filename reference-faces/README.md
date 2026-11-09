# Reference faces

The twenty old filename entries in this directory were removed from this branch after the audit showed that they were 36-byte `/api/images/...` text references, not JPEG/PNG image files.

**The original twenty photos are not available in this repository.** No actual user photo was deleted; the removed entries did not contain image pixels.

Import the real, consented reference photos into the native iOS app using PhotosPicker. The app:
- accepts up to 20 reference images;
- validates that exactly one face is detectable in each imported image;
- stores them in the app's private Application Support folder;
- lets the user select the source face and remove a local reference.

Do not commit personal face photographs, face embeddings, biometric templates, or other private identity data to this public GitHub repository. Common reference image formats are ignored by Git in this folder.