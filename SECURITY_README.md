# U-CogNet Security Framework

## Overview
U-CogNet implements comprehensive security measures to prevent self-modifying code and ensure safe AI operation.

## Security Layers

### 1. Sandboxing (Docker)
- **Read-only source code mounts**
- **Isolated execution environment**
- **Non-root user execution**
- **Limited system access**

### 2. Process Restrictions (AppArmor/SELinux)
- **File system access control**
- **Network restrictions**
- **System call limitations**
- **Process isolation**

### 3. Controlled Mutation Gateway
- **AI suggestions stored separately**
- **Human approval required**
- **Audit trail of all proposals**
- **Secure application process**

### 4. Version Control Security
- **Git hooks prevent unauthorized writes**
- **Immutable source code protection**
- **Signed commits and tags**

### 5. Cryptographic Auditing
- **Tamper-proof log entries**
- **Digital signatures**
- **Chain of trust verification**
- **Integrity monitoring**

## Installation

```bash
# Run the complete security setup
./install_security.sh

# Or step by step:
./setup_security.sh
```

## Usage

### Running Securely
```bash
# Run in Docker sandbox
docker run --rm \
  --security-opt apparmor=ucognet \
  --read-only \
  --tmpfs /tmp \
  -v /tmp/ucognet-data:/data \
  -v /tmp/ucognet-logs:/logs \
  -v $(pwd)/src:/app/src:ro \
  ucognet-secure
```

### Mutation Control
```python
from controlled_mutation_gateway import mutation_gateway

# AI suggests improvement
suggestion_id = mutation_gateway.suggest_mutation(
    "my_module",
    {"function": "optimize_algorithm"},
    "This will improve performance by 20%",
    0.85
)

# Human reviews and approves
mutation_gateway.approve_suggestion(suggestion_id, "human_reviewer")

# Apply approved changes
mutation_gateway.apply_approved_suggestion(suggestion_id)
```

### Security Monitoring
```python
from cryptographic_logger import secure_logger

# Log security events
event_id = secure_logger.log_event(
    "security_check",
    {"action": "code_integrity_check", "result": "passed"},
    "INFO"
)

# Verify log integrity
is_valid = secure_logger.verify_log_integrity()
```

## Security Principles

1. **Zero Trust**: Every operation is verified
2. **Defense in Depth**: Multiple security layers
3. **Human Oversight**: Critical decisions require approval
4. **Audit Everything**: Complete traceability
5. **Fail Safe**: Secure defaults, explicit permissions

## Monitoring

- **AppArmor**: Check `/var/log/syslog` for violations
- **Audit**: Check `/var/log/audit/audit.log` for file access
- **Cryptographic Logs**: Verify with `secure_logger.verify_log_integrity()`
- **Mutation Gateway**: Review pending suggestions with `mutation_gateway.list_pending_suggestions()`

## Emergency Procedures

If security violation detected:
1. Stop all U-CogNet processes
2. Review audit logs
3. Verify code integrity
4. Restore from known good backup
5. Investigate root cause

## Compliance

This security framework ensures:
- **No self-modification** without human approval
- **Complete audit trail** of all operations
- **Tamper-proof logging** with cryptographic signatures
- **Isolated execution** preventing system compromise