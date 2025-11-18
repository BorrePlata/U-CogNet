#!/bin/bash
# U-CogNet Security Setup Script
# Sets up AppArmor profiles and security measures

set -e

echo "🔒 Setting up U-CogNet Security Measures..."

# Install AppArmor profile
sudo apparmor_parser -r -W ucognet.apparmor

# Create audit rules for monitoring
sudo auditctl -w /mnt/c/Users/desar/Documents/Science/UCogNet/src/ -p wa -k ucognet_security
sudo auditctl -w /mnt/c/Users/desar/Documents/Science/UCogNet/*.py -p wa -k ucognet_security

# Set up immutable source code (if supported)
# sudo chattr +i /mnt/c/Users/desar/Documents/Science/UCogNet/src/*.py 2>/dev/null || true

echo "✅ Security measures applied successfully"
echo "📊 Monitoring active - check /var/log/audit/audit.log for violations"