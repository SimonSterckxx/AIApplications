from minio import Minio
import io

minio_url = "193.191.177.33:22555"
minio_user = "academic-weapons"
minio_passwd = "academic-weapons-peace"
bucket_name = "eyes4rescue-academic-weapons"

minioClient = Minio(
    minio_url,
    access_key=minio_user,
    secret_key=minio_passwd,
    secure=False
)

# Create empty folders by uploading zero-byte files
empty_file = io.BytesIO(b"")

# Simulate empty folders by uploading zero-byte files
minioClient.put_object(
    bucket_name, "movies/negative/.keep", empty_file, length=0)
minioClient.put_object(
    bucket_name, "movies/positive/.keep", empty_file, length=0)

print("Empty folders 'bucket1/' and 'bucket2/' created in the 'eyes4rescue-academic-weapons' bucket.")
