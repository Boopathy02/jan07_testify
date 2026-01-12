from datetime import datetime, timedelta
from jose import jwt

# IMPORTANT: This is a secret key for signing the JWTs.
# You should replace "your-secret-key" with a strong, randomly generated secret
# and ideally load it from an environment variable or a secure vault.
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt