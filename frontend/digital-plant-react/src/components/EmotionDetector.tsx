import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import styled from 'styled-components';
import Plant from './Plant';
import GradientBackground from './GradientBackground';
import MusicVisualizer from './MusicVisualizer';

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
    min-height: 100vh;
`;

const StatusMessage = styled.div`
    color: #666;
    margin: 2rem 0;
    font-size: 1.2rem;
`;

const EmotionBadge = styled(motion.div)<{ emotion: string }>`
    position: absolute;
    top: 20px;
    right: 20px;
    padding: 10px 20px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(5px);
    color: white;
    font-weight: bold;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const PlantContainer = styled(motion.div)`
    margin-top: 4rem;
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
                <AnimatePresence>
                    <EmotionBadge
                        emotion={plantData.emotion}
                        initial={{ x: 100, opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        exit={{ x: -100, opacity: 0 }}
                        key={plantData.emotion}
                    >
                        {plantData.emotion}
                    </EmotionBadge>
                </AnimatePresence>
                
                <PlantContainer
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.5 }}
                >
                    <EmotionText
                        initial={{ y: -20, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        transition={{ delay: 0.2 }}
                    >
                        Current Mood
                    </EmotionText>
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.5 }}
                    >
                        {plantData?.plant_image && (
                            <Plant 
                                plantImage={plantData.plant_image} 
                                emotion={plantData.emotion} 
                            />
                        )}
                    </motion.div>
                </PlantContainer>
            </Container>
            <MusicVisualizer emotion={plantData.emotion} />
        </>
    );
};

export default EmotionDetector;