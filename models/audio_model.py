import torch
import torch.nn as nn

from models.audio_extractor import AudioExtractor


# Must match training/train_audio.py exactly
CLASSES = {
    0: "real",
    1: "voiceclone",
    2: "audio_video_fake",
}


class AudioModel(nn.Module):
    """
    Wraps the Wav2Vec2-based AudioExtractor with the 3-class
    classifier head that was originally defined inline inside
    training/train_audio.py (nn.Linear(256, 3)).

    This class did not exist before -- audio_model.py was empty.
    It reconstructs the same architecture so a saved checkpoint
    from train_audio.py (extractor + classifier state dicts) can
    be loaded and reused for inference.
    """

    def __init__(self, num_classes=3):
        super().__init__()
        self.extractor = AudioExtractor()
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, audio):
        features = self.extractor(audio)
        logits = self.classifier(features)
        return logits

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device="cpu", num_classes=3):
        """
        Loads weights saved by training/train_audio.py, which saves
        a dict: {"extractor": ..., "classifier": ...}
        """
        model = cls(num_classes=num_classes).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.extractor.load_state_dict(checkpoint["extractor"])
        model.classifier.load_state_dict(checkpoint["classifier"])
        model.eval()
        return model
