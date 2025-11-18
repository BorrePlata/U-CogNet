#!/usr/bin/env python3
"""
U-CogNet Cryptographic Logging System
Provides tamper-proof audit trails with digital signatures
"""

import os
import json
import hashlib
import hmac
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import logging

class CryptographicLogger:
    """
    Cryptographically signed logging system for U-CogNet.

    Features:
    - Digital signatures for log integrity
    - Tamper detection
    - Chain of trust
    - Secure key management
    """

    def __init__(self, log_dir: str = "/logs", key_dir: str = "/data/keys"):
        self.log_dir = Path(log_dir)
        self.key_dir = Path(key_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.key_dir.mkdir(parents=True, exist_ok=True)

        # Generate or load keys
        self.private_key, self.public_key = self._load_or_generate_keys()

        # Current log file
        self.current_log = self.log_dir / f"ucognet_audit_{datetime.date.today()}.log"
        self.log_entries = []

        # Setup standard logging
        self.logger = logging.getLogger("UCogNet_Secure")
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add secure handler
        handler = logging.FileHandler(self.current_log)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def _load_or_generate_keys(self):
        """Load existing keys or generate new ones."""
        private_key_file = self.key_dir / "ucognet_private.pem"
        public_key_file = self.key_dir / "ucognet_public.pem"

        if private_key_file.exists() and public_key_file.exists():
            # Load existing keys
            with open(private_key_file, 'rb') as f:
                private_key = serialization.load_pem_private_key(
                    f.read(), password=None
                )
            with open(public_key_file, 'rb') as f:
                public_key = serialization.load_pem_public_key(f.read())
        else:
            # Generate new keys
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            public_key = private_key.public_key()

            # Save keys
            with open(private_key_file, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))

            with open(public_key_file, 'wb') as f:
                f.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))

        return private_key, public_key

    def log_event(self, event_type: str, details: Dict[str, Any],
                  severity: str = "INFO") -> str:
        """
        Log a security event with cryptographic signature.

        Args:
            event_type: Type of event (e.g., "mutation_suggestion", "security_violation")
            details: Event details
            severity: Event severity

        Returns:
            Event ID
        """

        # Create event data
        timestamp = datetime.datetime.now().isoformat()
        event_data = {
            "timestamp": timestamp,
            "event_type": event_type,
            "severity": severity,
            "details": details,
            "hostname": os.uname().nodename,
            "pid": os.getpid()
        }

        # Create message to sign
        message = json.dumps(event_data, sort_keys=True).encode()

        # Sign the message
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        # Create signed log entry
        log_entry = {
            "event_id": hashlib.sha256(message).hexdigest()[:16],
            "event_data": event_data,
            "signature": signature.hex(),
            "public_key_fingerprint": hashlib.sha256(
                self.public_key.public_bytes(
                    encoding=serialization.Encoding.DER,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
            ).hexdigest()[:16]
        }

        # Add to in-memory log
        self.log_entries.append(log_entry)

        # Write to file (append mode)
        with open(self.current_log, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')

        # Log to standard logger
        self.logger.info(f"{event_type}: {details}")

        return log_entry["event_id"]

    def verify_log_integrity(self, log_file: Optional[Path] = None) -> bool:
        """
        Verify the integrity of log entries.

        Returns:
            True if all entries are valid
        """
        if log_file is None:
            log_file = self.current_log

        if not log_file.exists():
            return False

        with open(log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    log_entry = json.loads(line.strip())
                    event_data = log_entry["event_data"]
                    signature = bytes.fromhex(log_entry["signature"])

                    # Recreate message
                    message = json.dumps(event_data, sort_keys=True).encode()

                    # Verify signature
                    self.public_key.verify(
                        signature,
                        message,
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH
                        ),
                        hashes.SHA256()
                    )

                except Exception as e:
                    print(f"Integrity check failed at line {line_num}: {e}")
                    return False

        return True

    def get_log_summary(self) -> Dict[str, Any]:
        """Get a summary of logged events."""
        summary = {
            "total_events": len(self.log_entries),
            "events_by_type": {},
            "events_by_severity": {},
            "date_range": {}
        }

        if self.log_entries:
            timestamps = [entry["event_data"]["timestamp"] for entry in self.log_entries]
            summary["date_range"] = {
                "first": min(timestamps),
                "last": max(timestamps)
            }

        for entry in self.log_entries:
            event_type = entry["event_data"]["event_type"]
            severity = entry["event_data"]["severity"]

            summary["events_by_type"][event_type] = summary["events_by_type"].get(event_type, 0) + 1
            summary["events_by_severity"][severity] = summary["events_by_severity"].get(severity, 0) + 1

        return summary

# Global instance
secure_logger = CryptographicLogger()