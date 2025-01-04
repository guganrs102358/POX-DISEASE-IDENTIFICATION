import React, { useState } from 'react';
import axios from 'axios';

const FileUpload = () => {
    const [image, setImage] = useState(null);
    const [prediction, setPrediction] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleFileChange = (e) => {
        setImage(e.target.files[0]);
        setPrediction('');
        setError('');
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!image) {
            alert("Please select an image first!");
            return;
        }

        const formData = new FormData();
        formData.append("file", image);

        setLoading(true);
        setError('');
        try {
            const response = await axios.post('http://localhost:5000/predict', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setPrediction(response.data.prediction);
        } catch (error) {
            console.error("Error uploading image:", error);
            setError(
                error.response && error.response.data.error
                    ? error.response.data.error
                    : "Error processing the image"
            );
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ fontFamily: 'Arial, sans-serif', padding: '20px', textAlign: 'center' }}>
            <h2>Image Prediction</h2>
            <form onSubmit={handleSubmit}>
                <input type="file" onChange={handleFileChange} accept="image/*" style={{ marginBottom: '10px' }} />
                <br />
                <button type="submit" disabled={loading || !image} style={{ padding: '5px 10px', marginTop: '10px' }}>
                    {loading ? 'Processing...' : 'Submit'}
                </button>
            </form>

            {error && (
                <div style={{ color: 'red', marginTop: '10px' }}>
                    <p>{error}</p>
                </div>
            )}

            {prediction && (
                <div style={{ color: 'green', marginTop: '10px' }}>
                    <p>Prediction: {prediction}</p>
                </div>
            )}
        </div>
    );
};

export default FileUpload;
