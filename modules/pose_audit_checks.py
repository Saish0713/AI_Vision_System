"""Small independent checks for the Stage 1 audit, without production data writes."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from modules.audit_how2sign_splits import distribution, metrics, normalize, probe


class AuditChecks(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize('Hello, WORLD!'), normalize('hello world'))
        self.assertEqual(normalize('Don’t!'), 'dont')
        self.assertNotEqual(normalize('go home'), normalize('stay home'))

    def test_invalid_distributions(self):
        for values in ([], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                distribution(values)

    def test_real_and_corrupt_video(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'clip.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 24))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((24, 32, 3), np.uint8))
            writer.release()
            result = probe(('test', path, 'fixture'))
            self.assertIsNone(result[3])
            self.assertEqual(result[2]['frames'], 3)
            self.assertAlmostEqual(result[2]['seconds'], 0.3)
            bad = Path(directory) / 'bad.mp4'
            bad.write_bytes(b'broken')
            self.assertIsNotNone(probe(('bad', bad, 'fixture'))[3])

    def test_metrics(self):
        good = metrics(['one two three four'], ['one two three four'])
        bad = metrics(['one two three four'], ['five six seven eight'])
        self.assertGreater(good['bleu'], 99.9)
        self.assertGreater(good['chrf'], 99.9)
        self.assertLess(bad['bleu'], good['bleu'])


if __name__ == '__main__':
    unittest.main()
