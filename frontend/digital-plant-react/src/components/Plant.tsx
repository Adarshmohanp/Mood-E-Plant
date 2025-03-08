import React from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';

interface Props {
    plantImage: string;
}

const PlantWrapper = styled.div`
    position: relative;
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
`;

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
`;

const PlantImage = styled.img`
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
`;

const Plant: React.FC<Props> = ({ plantImage }) => {
    return (
        <PlantWrapper>
            <PlantContainer
                animate={{ y: [0, -5, 0] }}
                transition={{ duration: 2, repeat: Infinity }}
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