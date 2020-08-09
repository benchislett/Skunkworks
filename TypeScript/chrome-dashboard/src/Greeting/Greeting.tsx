import React, { useEffect } from 'react';

import moment from 'moment';

import './Greeting.css';
import { Typography } from '@material-ui/core';

const quotesByGenre = require('./quotes.json');

const getTime = () => moment().format('hh:mm A');

function Clock() {
  const [time, setTime] = React.useState(getTime);

  useEffect(() => {
    const interval = setInterval(() => setTime(getTime()), 20000);
    return () => clearInterval(interval);
  }, []);

  return <Typography variant='h1'>{time}</Typography>;
}

interface QuoteProps {
  genres: string[];
}

function Quote({ genres }: QuoteProps) {
  const options = genres
    .filter((x) => quotesByGenre[x])
    .reduce((prev, next): any => [...prev, ...quotesByGenre[next]], []);
  let quote = '';
  do {
    quote = options[~~(Math.random() * (options.length - 1))];
  } while (
    quote[0].length > 80 ||
    quote[0].includes('my') ||
    quote[0].includes('me') ||
    quote[0].includes('I ')
  );

  let renderString = quote[0];
  if (quote[0].endsWith('.')) renderString = quote[0].slice(0, -1);

  return (
    <Typography variant='h4' style={{ color: 'white' }}>
      {renderString}
    </Typography>
  );
}

Quote.defaultProps = {
  genres: ['motivational', 'inspirational', 'journey', 'smile', 'happy']
};

function Greeting() {
  return (
    <div className='greeting-root'>
      <div className='greeting-time'>
        <Clock />
      </div>
      <div className='greeting-hello'>
        <Quote />
      </div>
    </div>
  );
}

export { Greeting };
