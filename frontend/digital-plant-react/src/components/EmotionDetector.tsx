import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import Plant from './Plant';
import GradientBackground from './GradientBackground';

// Type definitions
interface DetectionResult {
    emotion: string;
    confidence: number;
    plant_image: string;
}

interface WebcamResponse {
    results: DetectionResult[];
    error?: string;  // Add optional error property
}

// Styled components
const Container = styled.div`
    position: relative;
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
    z-index: 1;
`;

const StatusMessage = styled.div`
    color: #666;
    margin: 2rem 0;
    font-size: 1.2rem;
`;

const EmotionText = styled(motion.h2)`
    color: white;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    font-size: 2rem;
    margin-bottom: 2rem;
`;

const EmotionDetector: React.FC = () => {
    const [plantData, setPlantData] = useState<DetectionResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchWebcamData = async () => {
            try {
                const response = await fetch('http://localhost:8000/webcam-feed');
                if (!response.ok) {
                    throw new Error('Failed to fetch webcam data');
                }
                const data: WebcamResponse = await response.json();
                if (data.error) {
                    throw new Error(data.error);
                }
                if (data.results && data.results.length > 0) {
                    setPlantData(data.results[0]);
                    setError(null);
                } else {
                    setError('No face detected');
                }
            } catch (error) {
                console.error('Error:', error);
                setError(error instanceof Error ? error.message : 'Unknown error');
            }
        };

        const interval = setInterval(fetchWebcamData, 1000);
        return () => clearInterval(interval);
    }, []);

    if (error) return <StatusMessage>{error}</StatusMessage>;
    if (!plantData) return <StatusMessage>Waiting for emotion detection...</StatusMessage>;

    return (
        <>
            <GradientBackground emotion={plantData?.emotion || 'neutral'} />
            <Container>
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5 }}
                >
                    <EmotionText>Current Emotion: {plantData?.emotion}</EmotionText>
                    {plantData?.plant_image && (
                        <Plant plantImage={plantData.plant_image} />
                    )}
                </motion.div>
            </Container>
        </>
    );
};

export default EmotionDetector;