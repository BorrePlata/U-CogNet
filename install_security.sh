#!/bin/bash
# U-CogNet Complete Security Setup
# Implements comprehensive security measures for safe AI operation

set -e

echo "🛡️  U-CogNet Security Implementation"
echo "===================================="

# 1. Setup Docker sandboxing
echo "🐳 Setting up Docker sandboxing..."
docker build -t ucognet-secure .

# 2. Setup AppArmor profile
echo "🔒 Installing AppArmor security profile..."
sudo cp ucognet.apparmor /etc/apparmor.d/
sudo apparmor_parser -r /etc/apparmor.d/ucognet.apparmor

# 3. Setup audit monitoring
echo "📊 Configuring audit monitoring..."
sudo auditctl -w /mnt/c/Users/desar/Documents/Science/UCogNet/src/ -p wa -k ucognet_security
sudo auditctl -w /mnt/c/Users/desar/Documents/Science/UCogNet/*.py -p wa -k ucognet_security

# 4. Make source code immutable (optional, commented out for safety)
# echo "🔐 Making source code immutable..."
# sudo chattr +i /mnt/c/Users/desar/Documents/Science/UCogNet/src/*.py 2>/dev/null || true

# 5. Setup Git hooks
echo "🔗 Installing Git security hooks..."
chmod +x .git/hooks/pre-commit

# 6. Create data and logs directories
echo "📁 Creating secure directories..."
mkdir -p /tmp/ucognet-data
mkdir -p /tmp/ucognet-logs
chmod 755 /tmp/ucognet-data
chmod 755 /tmp/ucognet-logs

# 7. Generate cryptographic keys
echo "🔑 Generating cryptographic keys..."
python3 -c "
from cryptographic_logger import CryptographicLogger
logger = CryptographicLogger('/tmp/ucognet-logs', '/tmp/ucognet-data/keys')
logger.log_event('security_setup', {'action': 'initial_setup'}, 'INFO')
print('Keys generated and initial log created')
"

echo ""
echo "✅ U-CogNet Security Setup Complete!"
echo ""
echo "🚀 To run U-CogNet securely:"
echo "docker run --rm \\"
echo "  --security-opt apparmor=ucognet \\"
echo "  --read-only \\"
echo "  --tmpfs /tmp \\"
echo "  -v /tmp/ucognet-data:/data \\"
echo "  -v /tmp/ucognet-logs:/logs \\"
echo "  -v /mnt/c/Users/desar/Documents/Science/UCogNet/src:/app/src:ro \\"
echo "  ucognet-secure"
echo ""
echo "📋 Security Features Active:"
echo "  • Sandboxing with Docker"
echo "  • AppArmor process restrictions"
echo "  • Audit monitoring"
echo "  • Controlled mutation gateway"
echo "  • Git hooks for code protection"
echo "  • Cryptographic logging"
echo ""
echo "⚠️  Remember: AI can suggest improvements but cannot modify code directly"