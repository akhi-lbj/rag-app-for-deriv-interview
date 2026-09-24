# Account Security and Authentication Policy

## 1. Password Reset Policy
- Users can request a password reset via the login screen by entering their registered email address.
- **Attempt Limits**: Users are permitted a maximum of **3 password reset attempts per hour**.
- If a user exceeds 3 attempts within an hour, the account is temporarily locked for security purposes for **60 minutes**.
- Password reset links expire **15 minutes** after issuance.

## 2. Password Strength Requirements
- Passwords must be at least 12 characters in length.
- Passwords must contain at least one uppercase letter, one lowercase letter, one number, and one special character (!@#$%^&*).
- Passwords cannot match any of the previous 5 passwords used on the account.

## 3. Two-Factor Authentication (2FA)
- Two-factor authentication (2FA) via authenticator app (TOTP) is mandatory for all accounts handling financial transactions or API keys.
- SMS 2FA is supported as a secondary fallback only if TOTP has already been configured.
- Users are provided with 10 single-use recovery codes upon 2FA setup.

## 4. Account Lockout and Security Holds
- 5 consecutive failed login attempts will lock the account for 30 minutes.
- Changing a password or updating security settings automatically triggers a 24-hour security hold on all outward asset transfers.
