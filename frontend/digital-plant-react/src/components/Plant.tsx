import React from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';

interface Props {
    plantImage: string;
    emotion: string;
}

const PlantWrapper = styled.div`
    position: relative;
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
`;

const getEmotionAnimation = (emotion: string) => {
    switch (emotion.toLowerCase()) {
        case 'happy':
            return {
                animate: {
                    y: [0, -10, 0],
                    rotate: [-5, 5, -5],
                    scale: [1, 1.05, 1]
                },
                transition: {
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut"
                }
            };
        case 'sad':
            return {
                animate: {
                    y: [0, 5, 0],
                    rotate: [-2, 2, -2],
                    scale: [1, 0.95, 1]
                },
                transition: {
                    duration: 3,
                    repeat: Infinity,
                    ease: "easeInOut"
                }
            };
        case 'angry':
            return {
                animate: {
                    x: [-5, 5, -5],
                    rotate: [-3, 3, -3],
                },
                transition: {
                    duration: 0.5,
                    repeat: Infinity,
                    ease: "easeInOut"
                }
            };
        default:
            return {
                animate: {
                    scale: [1, 1.02, 1],
                },
                transition: {
                    duration: 4,
                    repeat: Infinity,
                    ease: "easeInOut"
                }
            };
    }
};

const PlantContainer = styled(motion.div)`
    width: 300px;
    height: 300px;
    margin: 0 auto;
    display: flex;
    justify-content: center;
    align-items: center;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(5px);
    border-radius: 15px;
    padding: 20px;
    overflow: hidden;
`;

const PlantImage = styled(motion.img)`
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
`;

const Plant: React.FC<Props> = ({ plantImage, emotion }) => {
    const emotionAnimation = getEmotionAnimation(emotion);

    return (
        <PlantWrapper>
            <PlantContainer
                {...emotionAnimation}
            >
                <PlantImage 
                    src={`data:image/png;base64,${plantImage}`} 
                    alt="Mood Plant"
                />
            </PlantContainer>
        </PlantWrapper>
    );
};

export default Plant;